#!/usr/bin/env python3
"""Build frozen Rubrics and trajectory evaluations from ShopSimulator rollouts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from shopping_grpo.evaluation.artifacts import (
    append_jsonl_fsync,
    atomic_jsonl_writer,
    index_jsonl,
    iter_jsonl,
    write_json_atomic,
)
from shopping_grpo.evaluation.blind_guard import guard_blind_final
from shopping_grpo.evaluation.comparison import compare_evaluation_runs
from shopping_grpo.evaluation.contracts import (
    rubric_ids,
    validate_judge_result,
    validate_rubric_bundle,
)
from shopping_grpo.evaluation.metrics import compute_deterministic_metrics
from shopping_grpo.evaluation.prompts import (
    RUBRIC_CURATOR_PROMPT_VERSION,
    TRAJECTORY_JUDGE_PROMPT_VERSION,
    build_rubric_curator_messages,
    build_trajectory_judge_messages,
)
from shopping_grpo.evaluation.results import (
    assemble_task_evaluation,
    build_not_judged_result,
    summarize_evaluations,
)
from shopping_grpo.evaluation.rubric import (
    extract_rubric_candidates,
    materialize_rubric_bundle,
    stable_hash,
)
from shopping_grpo.evaluation.task_facts import task_facts_from_environment
from shopping_grpo.evaluation.model_client import client_from_environment
from shopping_grpo.evaluation.trajectory import normalize_trajectory

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHOPSIM_ROOT = ROOT / "environments/ShopSimulator/shop_env"
PREPROCESSED_SCHEMA = "shopping-preprocessed-trajectory-v1"
JUDGE_REQUEST_SCHEMA = "shopping-judge-request-v2"
CURATOR_PROMPT_VERSION = RUBRIC_CURATOR_PROMPT_VERSION
JUDGE_PROMPT_VERSION = TRAJECTORY_JUDGE_PROMPT_VERSION
ENVIRONMENT_VERSION = "shopsimulator-environment-v2.1"

JUDGE_MODEL = "deepseek-v4.1-flash"


def _report(**fields) -> None:
    print(json.dumps(fields, ensure_ascii=False, sort_keys=True))


def _guard(paths, allowed: bool) -> None:
    guard_blind_final([Path(p) for p in paths], allowed=allowed)


def _resume_index(path: Path, *, key: str) -> dict:
    return index_jsonl(path, key=key) if path.is_file() else {}


def _client(args, *, model: str, max_tokens: int):
    return client_from_environment(
        model=model,
        max_tokens=max_tokens,
        timeout=args.timeout,
        retries=args.retries,
        response_format_json=args.json_format,
        thinking=args.thinking,
        reasoning_effort=args.reasoning_effort,
    )


# --------------------------------------------------------------------------- facts


def stage_facts(args: argparse.Namespace) -> None:
    _guard([args.tasks], args.allow_blind_final)
    root = args.shopsim_root.resolve()
    if not (root / "web_agent_site").is_dir():
        raise SystemExit("--shopsim-root 必须指向 ShopSimulator/shop_env")
    try:
        os.environ["SHOP_ENVIRONMENT_VERSION"] = ENVIRONMENT_VERSION
        sys.path.insert(0, str(root))
        from web_agent_site.engine.engine import load_products
        from web_agent_site.engine.goal import get_goals
        from web_agent_site.utils import DEFAULT_FILE_PATH
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"facts 阶段需要 ShopSimulator 依赖（{exc.name}）；"
            f"请用 {DEFAULT_SHOPSIM_ROOT.parent}/.venv-shopsim/bin/python 运行"
        ) from None
    products, product_item_dict, prices, _ = load_products(
        DEFAULT_FILE_PATH, num_products=None, human_goals=None
    )
    goals = get_goals(products, prices)
    rows = task_facts_from_environment(
        task_ids=[int(row["task_id"]) for row in iter_jsonl(args.tasks)],
        goals=goals,
        product_item_dict=product_item_dict,
    )
    with atomic_jsonl_writer(args.output, force=args.force) as write:
        for row in rows:
            write(row)
    _report(command="facts", tasks=len(rows), output=str(args.output))


# ---------------------------------------------------------------------- preprocess


def stage_preprocess(args: argparse.Namespace) -> None:
    _guard([args.raw], args.allow_blind_final)
    written = 0
    seen = set()
    with atomic_jsonl_writer(args.output, force=args.force) as write:
        for raw in iter_jsonl(args.raw):
            if args.limit is not None and written >= args.limit:
                break
            normalized = normalize_trajectory(raw)
            trajectory_id = normalized["trajectory_id"]
            if not trajectory_id:
                raise SystemExit("raw trajectory 缺少 trajectory_id")
            if trajectory_id in seen:
                raise SystemExit(f"重复 trajectory_id={trajectory_id!r}")
            seen.add(trajectory_id)
            write(
                {
                    "schema_version": PREPROCESSED_SCHEMA,
                    "task_id": normalized["task_id"],
                    "trajectory_id": trajectory_id,
                    "normalized_trajectory": normalized,
                    "deterministic_metrics": compute_deterministic_metrics(
                        normalized
                    ),
                }
            )
            written += 1
    _report(command="preprocess", written=written, output=str(args.output))


# ---------------------------------------------------------------- rubric candidates


def stage_rubric_candidates(args: argparse.Namespace) -> None:
    _guard([args.task_facts], args.allow_blind_final)
    written = 0
    with atomic_jsonl_writer(args.output, force=args.force) as write:
        for task_facts in iter_jsonl(args.task_facts):
            write(extract_rubric_candidates(task_facts))
            written += 1
    _report(
        command="rubric-candidates", written=written, output=str(args.output)
    )




# -------------------------------------------------------------------------- rubric


def stage_rubric(args: argparse.Namespace) -> None:
    _guard([args.task_facts, args.candidates], args.allow_blind_final)
    task_facts = index_jsonl(args.task_facts, key="task_id")
    candidates = index_jsonl(args.candidates, key="task_id")
    if set(task_facts) != set(candidates):
        raise SystemExit("task facts 与 candidates 的 task_id 必须完全一致")
    if args.output.is_file() and not (args.resume or args.force):
        raise SystemExit(f"输出已存在：{args.output}；用 --resume 或 --force")
    existing = _resume_index(args.output, key="task_id") if args.resume else {}
    if set(existing) - set(task_facts):
        raise SystemExit("cached Rubric 包含本次任务集之外的 task_id")
    for task_id, bundle in existing.items():
        validate_rubric_bundle(bundle, expected_task_id=int(task_id))
        generation = bundle.get("generation") or {}
        if (
            generation.get("curator_model") != args.curator_model
            or generation.get("curator_prompt_version") != CURATOR_PROMPT_VERSION
            or bundle.get("rubric_version") != args.rubric_version
            or generation.get("task_data_hash") != str(task_facts[task_id].get("task_data_hash"))
            or generation.get("query_hash") != str(task_facts[task_id].get("query_hash"))
            or generation.get("extractor_version") != candidates[task_id].get("extractor_version")
        ):
            raise SystemExit(
                f"cached Rubric 与本次运行不兼容（task_id={task_id}）：模型、prompt、版本或数据已变化"
            )
    if not args.resume:
        args.output.unlink(missing_ok=True)
    written = 0
    failures = []
    client = None
    for task_id in sorted(task_facts):
        if task_id in existing:
            continue
        if args.limit is not None and written >= args.limit:
            break
        facts = task_facts[task_id]
        bundle_candidates = candidates[task_id]
        if client is None:
            client = _client(args, model=args.curator_model, max_tokens=args.max_tokens)
        messages = build_rubric_curator_messages(
            task_id=int(task_id), query=str(facts["query"]),
            candidates=bundle_candidates["candidates"],
        )
        try:
            completion = client.complete_json(messages)
            bundle = materialize_rubric_bundle(
                task_facts=facts, candidates=bundle_candidates,
                curator_response=completion["result"],
                curator_model=args.curator_model,
                curator_prompt_version=CURATOR_PROMPT_VERSION,
                rubric_version=args.rubric_version,
            )
        except Exception as exc:
            failures.append({"task_id": task_id, "error": f"{type(exc).__name__}: {exc}"})
            continue
        append_jsonl_fsync(args.output, bundle)
        written += 1
    _report(command="rubric", model=args.curator_model, written=written,
            cached=len(existing), failed=len(failures), output=str(args.output))
    for failure in failures:
        print(json.dumps(failure, ensure_ascii=False, sort_keys=True))
    if failures:
        raise SystemExit(f"{len(failures)} 题 Rubric 生成失败；用 --resume 补跑")


# -------------------------------------------------------------------- judge inputs


def stage_judge_inputs(args: argparse.Namespace) -> None:
    _guard([args.preprocessed, args.rubrics], args.allow_blind_final)
    rubrics = index_jsonl(args.rubrics, key="task_id")
    written = 0
    with atomic_jsonl_writer(args.output, force=args.force) as write:
        for row in iter_jsonl(args.preprocessed):
            if row.get("schema_version") != PREPROCESSED_SCHEMA:
                raise SystemExit("不支持的 preprocessed schema")
            task_id = int(row["task_id"])
            rubric = rubrics.get(task_id)
            if rubric is None:
                raise SystemExit(f"task_id={task_id} 缺少 Rubric")
            normalized = row["normalized_trajectory"]
            metrics = row["deterministic_metrics"]
            validity = metrics.get("validity") or {}
            request = {
                "schema_version": JUDGE_REQUEST_SCHEMA,
                "task_id": task_id,
                "trajectory_id": row["trajectory_id"],
                "prompt_version": JUDGE_PROMPT_VERSION,
            }
            if validity.get("infrastructure_invalid"):
                request["judge_required"] = False
                request["not_judged_result"] = build_not_judged_result(
                    task_id=task_id,
                    trajectory_id=row["trajectory_id"],
                    reason="infrastructure_invalid",
                )
            else:
                request["judge_required"] = True
                request["rubric_ids"] = rubric_ids(rubric)
                request["allowed_event_ids"] = [
                    event["event_id"]
                    for event in normalized.get("events") or []
                    if isinstance(event, dict) and event.get("event_id")
                ]
                request["messages"] = build_trajectory_judge_messages(
                    normalized=normalized,
                    rubric_bundle=rubric,
                    deterministic_metrics=metrics,
                )
            write(request)
            written += 1
    _report(command="judge-inputs", written=written, output=str(args.output))


# --------------------------------------------------------------------------- judge


def stage_judge(args: argparse.Namespace) -> None:
    _guard([args.requests], args.allow_blind_final)
    by_trajectory = index_jsonl(args.requests, key="trajectory_id")
    requests = list(by_trajectory.values())
    for request in requests:
        if request.get("schema_version") != JUDGE_REQUEST_SCHEMA or request.get("prompt_version") != JUDGE_PROMPT_VERSION:
            raise SystemExit("不支持的 Judge request schema 或 prompt 版本")
    if args.output.is_file() and not (args.resume or args.force):
        raise SystemExit(f"输出已存在：{args.output}；用 --resume 或 --force")
    existing = _resume_index(args.output, key="trajectory_id") if args.resume else {}
    for trajectory_id, result in existing.items():
        request = by_trajectory.get(trajectory_id)
        if request is None:
            raise SystemExit(f"cached Judge 包含本次请求之外的 trajectory_id={trajectory_id}")
        validate_judge_result(
            result,
            rubric_ids=request.get("rubric_ids", []),
            expected_task_id=int(request["task_id"]),
            expected_trajectory_id=trajectory_id,
            allowed_event_ids=request.get("allowed_event_ids", []),
        )
        if (
            result.get("judge_model") != JUDGE_MODEL
            or result.get("judge_prompt_version") != JUDGE_PROMPT_VERSION
            or result.get("judge_request_hash") != stable_hash(request)
        ):
            raise SystemExit(
                f"cached Judge 与本次运行不兼容（trajectory_id={trajectory_id}）："
                "模型、prompt 或输入已变化；请用 --force 重跑"
            )
    if not args.resume:
        args.output.unlink(missing_ok=True)
    pending = [
        request
        for request in requests
        if str(request["trajectory_id"]) not in existing
    ]
    if args.limit is not None:
        pending = pending[: args.limit]

    def run(request: dict) -> dict:
        trajectory_id = str(request["trajectory_id"])
        task_id = int(request["task_id"])
        if request.get("judge_required") is False:
            result = dict(request["not_judged_result"])
        else:
            client = _client(args, model=JUDGE_MODEL, max_tokens=args.max_tokens)
            completion = client.complete_json(request["messages"])
            result = validate_judge_result(
                completion["result"],
                rubric_ids=request["rubric_ids"],
                expected_task_id=task_id,
                expected_trajectory_id=trajectory_id,
                allowed_event_ids=request["allowed_event_ids"],
            )
        result["judge_model"] = JUDGE_MODEL
        result["judge_prompt_version"] = JUDGE_PROMPT_VERSION
        result["judge_request_hash"] = stable_hash(request)
        return result

    written = 0
    failures = []
    if pending:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run, request): request for request in pending}
            for future in as_completed(futures):
                request = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    failures.append(
                        {
                            "task_id": request.get("task_id"),
                            "trajectory_id": request.get("trajectory_id"),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                append_jsonl_fsync(args.output, result)
                written += 1
    _report(
        command="judge",
        model=JUDGE_MODEL,
        cached=len(existing),
        written=written,
        failed=len(failures),
        output=str(args.output),
    )
    for failure in failures:
        print(json.dumps(failure, ensure_ascii=False, sort_keys=True))
    if failures:
        raise SystemExit(
            f"{len(failures)} 条 Judge 请求失败；已写出的结果保留，"
            "用 --resume 重跑即可只补这些条目"
        )


# ------------------------------------------------------------------------- assemble


def stage_assemble(args: argparse.Namespace) -> None:
    _guard(
        [args.preprocessed, args.rubrics, args.judges, args.tasks],
        args.allow_blind_final,
    )
    expected_task_ids = [int(row["task_id"]) for row in iter_jsonl(args.tasks)]
    expected = set(expected_task_ids)
    rubrics = index_jsonl(args.rubrics, key="task_id", allowed_keys=expected)
    judges = index_jsonl(args.judges, key="trajectory_id")
    actor = {"actor_id": args.actor_label, "model": args.actor_label}
    evaluations = []
    seen = set()
    for row in iter_jsonl(args.preprocessed):
        if row.get("schema_version") != PREPROCESSED_SCHEMA:
            raise SystemExit("不支持的 preprocessed schema")
        task_id = int(row["task_id"])
        if task_id not in expected:
            raise SystemExit(f"preprocessed 出现预期外的 task_id={task_id}")
        if task_id in seen:
            raise SystemExit(f"每个 task 只允许一条轨迹；重复 {task_id}")
        seen.add(task_id)
        rubric = rubrics.get(task_id)
        if rubric is None:
            raise SystemExit(f"task_id={task_id} 缺少 Rubric")
        judge = judges.get(row["trajectory_id"])
        if judge is None:
            validity = row["deterministic_metrics"].get("validity") or {}
            if not validity.get("infrastructure_invalid"):
                raise SystemExit(
                    f"trajectory_id={row['trajectory_id']} 缺少 Judge 结果"
                )
            judge = build_not_judged_result(
                task_id=task_id,
                trajectory_id=row["trajectory_id"],
                reason="infrastructure_invalid",
            )
        evaluations.append(
            assemble_task_evaluation(
                actor=actor,
                normalized_trajectory=row["normalized_trajectory"],
                deterministic_metrics=row["deterministic_metrics"],
                rubric_bundle=rubric,
                judge_result=judge,
            )
        )
    with atomic_jsonl_writer(args.output, force=args.force) as write:
        for evaluation in evaluations:
            write(evaluation)
    summary = summarize_evaluations(
        expected_task_ids=expected_task_ids,
        evaluations=evaluations,
    )
    write_json_atomic(args.summary, summary, force=args.force)
    _report(
        command="assemble",
        evaluations=len(evaluations),
        expected_tasks=len(expected_task_ids),
        output=str(args.output),
        summary=str(args.summary),
    )


# -------------------------------------------------------------------------- compare


def stage_compare(args: argparse.Namespace) -> None:
    runs = {}
    paths = [args.tasks]
    for specification in args.model_evaluations:
        if "=" not in specification:
            raise SystemExit("--model-evaluations 需用 LABEL=PATH")
        label, raw_path = specification.split("=", 1)
        label = label.strip()
        if not label or label in runs:
            raise SystemExit(f"非法或重复的 label {label!r}")
        path = Path(raw_path)
        paths.append(path)
        runs[label] = list(iter_jsonl(path))
    _guard(paths, args.allow_blind_final)
    comparison = compare_evaluation_runs(
        expected_task_ids=[int(row["task_id"]) for row in iter_jsonl(args.tasks)],
        runs=runs,
    )
    write_json_atomic(args.output, comparison, force=args.force)
    _report(
        command="compare",
        models=list(runs),
        output=str(args.output),
    )


# ----------------------------------------------------------------------------- CLI


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    """Repeat on every subcommand: global flags must be usable after the verb."""
    parser.add_argument(
        "--allow-blind-final",
        action="store_true",
        help="显式确认处理 Final-200 Clean 冻结集；不传则拒绝",
    )
    parser.add_argument("--force", action="store_true", help="允许覆盖已有输出")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="校验已有输出并只补缺失条目（rubric / judge）",
    )


def _llm_arguments(parser: argparse.ArgumentParser, *, default_tokens: int) -> None:
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=default_tokens)
    parser.add_argument("--json-format", action="store_true")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--reasoning-effort", default="high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    command = sub.add_parser("facts", help="从 ShopSimulator 导出私有 TaskFacts")
    command.add_argument("--shopsim-root", type=Path, default=DEFAULT_SHOPSIM_ROOT)
    command.add_argument("--tasks", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    _common_arguments(command)
    command.set_defaults(handler=stage_facts)

    command = sub.add_parser("preprocess", help="规范化 raw rollout 并算确定性指标")
    command.add_argument("--raw", type=Path, required=True)
    command.add_argument("--limit", type=int)
    command.add_argument("--output", type=Path, required=True)
    _common_arguments(command)
    command.set_defaults(handler=stage_preprocess)

    command = sub.add_parser("rubric-candidates", help="从 TaskFacts 提取候选约束")
    command.add_argument("--task-facts", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    _common_arguments(command)
    command.set_defaults(handler=stage_rubric_candidates)

    command = sub.add_parser("rubric", help="候选 -> 冻结 Rubric Bundle")
    command.add_argument("--task-facts", type=Path, required=True)
    command.add_argument("--candidates", type=Path, required=True)
    command.add_argument("--rubric-version", default="task-rubric-v1")
    command.add_argument("--limit", type=int)
    _llm_arguments(command, default_tokens=2048)
    command.add_argument("--curator-model", default="deepseek-v4.1-flash")
    command.add_argument("--output", type=Path, required=True)
    _common_arguments(command)
    command.set_defaults(handler=stage_rubric)

    command = sub.add_parser("judge-inputs", help="构造 Judge 请求（含输入隔离）")
    command.add_argument("--preprocessed", type=Path, required=True)
    command.add_argument("--rubrics", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    _common_arguments(command)
    command.set_defaults(handler=stage_judge_inputs)

    command = sub.add_parser("judge", help="调用 Flash Judge 模型并校验结果")
    command.add_argument("--requests", type=Path, required=True)
    command.add_argument("--workers", type=int, default=3)
    command.add_argument("--limit", type=int)
    _llm_arguments(command, default_tokens=4096)
    command.add_argument("--output", type=Path, required=True)
    _common_arguments(command)
    command.set_defaults(handler=stage_judge)

    command = sub.add_parser("assemble", help="拼装四面板并汇总")
    command.add_argument("--preprocessed", type=Path, required=True)
    command.add_argument("--rubrics", type=Path, required=True)
    command.add_argument("--judges", type=Path, required=True)
    command.add_argument("--tasks", type=Path, required=True)
    command.add_argument("--actor-label", default="model")
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--summary", type=Path, required=True)
    _common_arguments(command)
    command.set_defaults(handler=stage_assemble)

    command = sub.add_parser("compare", help="多模型按 task_id 配对比较")
    command.add_argument("--tasks", type=Path, required=True)
    command.add_argument("--model-evaluations", nargs="+", required=True)
    command.add_argument("--output", type=Path, required=True)
    _common_arguments(command)
    command.set_defaults(handler=stage_compare)

    args = parser.parse_args()
    if getattr(args, "limit", None) is not None and args.limit < 1:
        raise SystemExit("--limit 必须为正")
    if args.resume and args.force:
        raise SystemExit("--resume 与 --force 互斥")
    return args


def main() -> None:
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
