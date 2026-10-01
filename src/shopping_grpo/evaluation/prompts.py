"""Frozen prompts and Judge-safe payload renderers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy

from shopping_grpo.evaluation.contracts import (
    ERROR_TAXONOMY,
    JUDGE_DIMENSIONS,
    JUDGE_SCHEMA_VERSION,
    validate_rubric_bundle,
)
from shopping_grpo.evaluation.trajectory import NORMALIZED_TRAJECTORY_VERSION

RUBRIC_CURATOR_PROMPT_VERSION = "rubric-curator-v1-draft-r4"
TRAJECTORY_JUDGE_PROMPT_VERSION = "trajectory-judge-v1-draft-r4"
_JUDGE_VISIBLE_ERROR_TAXONOMY = ERROR_TAXONOMY - {
    "reward_rubric_disagreement",
    "infrastructure_invalid",
}

RUBRIC_CURATOR_SYSTEM_PROMPT = """\
你是当前 Shopping Agent 项目的需求 Rubric 整理器，不是自由生成需求的助手。

你只能从输入的 candidates 中选择用户 Query 确实表达的约束，并做简短自然语言化。
严禁新增 candidate_id，严禁修改候选的底层字段、操作符或期望值，严禁把目标商品的
全部属性自动视为用户需求。

先保证完整覆盖 Query 中每一个彼此独立的明确要求，再做最小化和去重。“最小”只表示
同义或上下位重复要求不重复计分，绝不表示少选要求。每个 candidate 的
selection_guidance 是强制选择规则。

每条选择必须有 Query 原文直接支持：
- 泛化的目标商品属性不能仅凭常识或商品字段入选；
- 同一用户要求已经由更具体的 option、规格或价格候选覆盖时，不再拆成多个含义重叠
  的泛化 core_function；
- 品类词不能因为碰巧等于目标商品 brand 而被选成品牌约束；
- “适用于/兼容某品牌”是兼容性要求，不代表所购商品自身必须属于该品牌；
- Query 明确给出商品类型时，品类本身是一条独立要求，应和功能、规格分别保留；
- 组合 option 可以承载 Query 明确要求的多个规格，但不能借机加入用户未要求的品牌、
  型号、数量或实质规格；
- 每条入选约束必须提供 Query 中连续、逐字存在且能独立支持该约束的非空原文片段。

例：Query “推荐移动电源”中的“移动电源”是品类，不是品牌；Query “适用于海信
电视的回音壁”要求兼容海信电视，不要求回音壁品牌为海信。

hard/soft 规则：
- 明确品类、明确预算上限、否定要求、指定规格或选项属于 hard；
- “优先、最好、倾向、左右”等偏好属于 soft；
- candidates 中的 hardness_hint 只是代码初筛提示；最终 hard/soft 必须服从 Query
  原文措辞，“最好”等明确偏好不能被强制改成 hard；
- 无法可靠判断时使用 needs_review，不要强行二选一。

只输出一个 JSON 对象：
{
  "selected_constraints": [
    {
      "candidate_id": "c0001",
      "description": "非空、简短、人类可读且不扩写的新描述",
      "hardness": "hard | soft | needs_review",
      "query_quote": "非空且逐字存在于 Query 的连续原文",
      "selection_reason": "非空；说明该候选为何由这段原文直接支持"
    }
  ]
}
不要输出 Markdown、解释性前后缀或任何额外字段。"""

TRAJECTORY_JUDGE_SYSTEM_PROMPT = f"""\
你是当前 Shopping Agent / ShopSimulator 项目的离线轨迹 Judge。

你必须只依据输入中的 actor_visible_trajectory 评价 Actor 的行为。不得假设你能看到
audit raw_observation、Gold 商品私有字段或未展示给 Actor 的候选。输入不会包含
Environment Reward、Reward 分项或代码判定的任务成功结论；这些结果由独立面板
负责，不能由你推断、覆盖或改写。

逐条需求状态只能是 satisfied、violated、unknown、not_applicable。没有可见证据时
使用 unknown。每项判断尽量引用真实 event_id；不得伪造不存在的 event_id。

五个维度分别打 0、1、2 分，不加权、不计算总分：
- search_strategy：搜索是否覆盖品类和关键条件，改写是否有效，是否机械重复；
- candidate_utilization：是否利用可见的高匹配候选，比较是否必要且不过度；
- evidence_verification：购买前是否核验关键属性、规格和最终价格；
- decision_quality：最终选择、规格和购买/放弃决策是否合理；
- termination_efficiency：是否过早购买/放弃、无效探索或耗尽步骤。

错误类型必须从输入提供的 frozen_error_taxonomy 中选择。primary 只选一个
最能解释轨迹失败或低分的根因；secondary 最多两个次要错误，不得与
primary 重复。没有明显错误时 primary 使用 null，secondary 必须为空列表。
errors.evidence_event_ids 必须引用能支持归因的真实轨迹事件。
只输出 JSON，不输出 Markdown。
schema_version 必须是 {JUDGE_SCHEMA_VERSION}。禁止输出 total_score、overall_score
或任何综合分。"""


def build_rubric_curator_messages(
    *,
    task_id: int,
    query: str,
    candidates: list[Mapping],
) -> list[dict]:
    """Build an OpenAI-compatible Flash request without hidden free-form data."""

    payload = {
        "task_id": int(task_id),
        "query": str(query),
        "candidates": deepcopy(candidates),
    }
    return [
        {"role": "system", "content": RUBRIC_CURATOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
        },
    ]


def actor_visible_trajectory(normalized: Mapping) -> dict:
    """Strip all audit-only content before rendering a Judge request."""

    if normalized.get("schema_version") != NORMALIZED_TRAJECTORY_VERSION:
        raise ValueError("unsupported normalized trajectory schema")
    events = []
    allowed_event_fields = (
        "event_id",
        "event_type",
        "action_attempt_id",
        "executed_step_id",
        "assistant_text",
        "tool_call_id",
        "tool_name",
        "parameters",
        "tool_call_parse_error",
        "guard_reason",
        "guard_consecutive_count",
        "latest_observation_truncated",
        "env_action",
        "actor_visible_observation",
        "done",
        "step_error",
    )
    for event_value in normalized.get("events") or []:
        if not isinstance(event_value, Mapping):
            continue
        event = {
            key: deepcopy(event_value[key])
            for key in allowed_event_fields
            if key in event_value
        }
        events.append(event)
    return {
        "trajectory_id": normalized.get("trajectory_id"),
        "task_id": normalized.get("task_id"),
        "status": normalized.get("status"),
        "done": normalized.get("done"),
        "events": events,
    }


def sanitize_terminal_for_judge(terminal: Mapping) -> dict:
    """Expose only neutral lifecycle flags, never Reward-derived conclusions."""

    if not isinstance(terminal, Mapping):
        terminal = {}
    return {
        "done": bool(terminal.get("done")),
        "over": bool(terminal.get("over")),
    }


def judge_visible_metrics(deterministic_metrics: Mapping) -> dict:
    """Remove Reward/outcome and validity conclusions before LLM judging."""

    if not isinstance(deterministic_metrics, Mapping):
        deterministic_metrics = {}
    allowed_sections = (
        "actions_and_efficiency",
        "repetition",
        "legality",
        "context",
    )
    return {
        section: deepcopy(deterministic_metrics.get(section) or {})
        for section in allowed_sections
    }


def build_trajectory_judge_messages(
    *,
    normalized: Mapping,
    rubric_bundle: Mapping,
    deterministic_metrics: Mapping,
) -> list[dict]:
    """Build one Judge request with exactly the evidence the Actor could use."""

    rubric = validate_rubric_bundle(
        rubric_bundle,
        expected_task_id=int(normalized["task_id"]),
    )
    dimensions = {
        name: {"allowed_scores": [0, 1, 2]}
        for name in JUDGE_DIMENSIONS
    }
    payload = {
        "task_id": normalized["task_id"],
        "trajectory_id": normalized["trajectory_id"],
        "query": normalized.get("actor_query") or rubric["query"],
        "rubric": rubric["rubrics"],
        "dimension_spec": dimensions,
        "frozen_error_taxonomy": sorted(
            _JUDGE_VISIBLE_ERROR_TAXONOMY
        ),
        "actor_visible_trajectory": actor_visible_trajectory(normalized),
        "terminal_state": sanitize_terminal_for_judge(
            normalized.get("terminal") or {}
        ),
        "judge_visible_metrics": judge_visible_metrics(
            deterministic_metrics
        ),
        "required_output": {
            "schema_version": JUDGE_SCHEMA_VERSION,
            "task_id": normalized["task_id"],
            "trajectory_id": normalized["trajectory_id"],
            "judge_status": "valid | invalid | not_judged",
            "rubric_assessments": [
                {
                    "rubric_id": "每条输入 rubric_id 恰好一次",
                    "status": "satisfied | violated | unknown | not_applicable",
                    "reason": "简短理由",
                    "evidence_event_ids": ["e0001"],
                }
            ],
            "dimension_scores": {
                name: {
                    "score": "0 | 1 | 2",
                    "reason": "简短理由",
                    "evidence_event_ids": ["e0001"],
                }
                for name in JUDGE_DIMENSIONS
            },
            "errors": {
                "primary": "一个 frozen_error_taxonomy 值或 null",
                "secondary": ["最多两个不同的次要错误"],
                "evidence_event_ids": ["支持主次错误归因的 event_id"],
            },
            "overall_diagnosis": "简短整体诊断",
        },
    }
    return [
        {"role": "system", "content": TRAJECTORY_JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
        },
    ]
