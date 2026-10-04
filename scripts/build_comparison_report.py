#!/usr/bin/env python3
"""Build a self-contained comparison report for fixed-benchmark evaluation runs."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from shopping_grpo.evaluation.summary import REWARD_V3_TYPES


ROOT = Path(__file__).resolve().parents[1]
HISTOGRAM_EDGES = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.000001]


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _histogram(values: list[float]) -> list[int]:
    counts = [0] * (len(HISTOGRAM_EDGES) - 1)
    for value in values:
        for index, (lower, upper) in enumerate(zip(HISTOGRAM_EDGES, HISTOGRAM_EDGES[1:])):
            if lower <= value < upper:
                counts[index] += 1
                break
    return counts


def _reward_type(row: dict) -> str:
    detail = (row.get("terminal_result") or {}).get("reward_detail") or {}
    reward_type = detail.get("reward_type")
    return reward_type if reward_type in REWARD_V3_TYPES else "unknown"


def _model_data(run_dir: Path) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    expected_ids = [int(task_id) for task_id in summary["expected_task_ids"]]
    expected_set = set(expected_ids)
    total = int(summary["expected_tasks"])
    if total != len(expected_ids) or len(expected_set) != total:
        raise ValueError(f"inconsistent expected tasks in {run_dir}")
    by_task = {
        int(row["task_id"]): row
        for row in _load_jsonl(run_dir / "trajectories.jsonl")
        if int(row["task_id"]) in expected_set
    }
    rows = list(by_task.values())
    rewards = [float(row["final_reward"]) for row in rows]
    protocol = summary.get("protocol") or {}
    success_ids = [int(task_id) for task_id in summary["strict_success_task_ids"]]
    missing_ids = sorted(expected_set - set(by_task))
    return {
        "key": run_dir.name,
        "label": run_dir.name,
        "name": protocol.get("model") or run_dir.name,
        "report": f"{run_dir.name}/report.html",
        "tasks": total,
        "expected_task_ids": sorted(expected_ids),
        "recorded": len(rows),
        "missing": len(missing_ids),
        "missing_task_ids": missing_ids,
        "successes": len(success_ids),
        "success_rate": len(success_ids) / total if total else 0.0,
        "purchase_rate": float(summary["purchase_success_rate"]),
        "reward_valid_rate": float(summary["reward_valid_rate"]),
        "average_steps": float(summary["average_steps"]) if rows else None,
        "outcomes": dict(Counter(_reward_type(row) for row in rows)),
        "statuses": dict(Counter(row.get("status", "unknown") for row in rows)),
        "guards": summary["guard_reason_counts"],
        "success_ids": success_ids,
        "reward": {
            "count": len(rewards),
            "mean": statistics.fmean(rewards) if rewards else None,
            "median": statistics.median(rewards) if rewards else None,
            "stddev": statistics.pstdev(rewards) if rewards else None,
            "min": min(rewards) if rewards else None,
            "q1": _quantile(rewards, 0.25) if rewards else None,
            "q3": _quantile(rewards, 0.75) if rewards else None,
            "max": max(rewards) if rewards else None,
            "histogram": _histogram(rewards),
        },
    }


def build_comparison_data(evaluation_dir: Path) -> dict:
    run_dirs = sorted(
        path
        for path in evaluation_dir.iterdir()
        if path.is_dir()
        and (path / "summary.json").is_file()
        and (path / "trajectories.jsonl").is_file()
    )
    models = [_model_data(path) for path in run_dirs]
    if not models:
        raise ValueError(f"no evaluation runs found under {evaluation_dir}")
    all_task_ids = set(models[0]["expected_task_ids"])
    if any(set(model["expected_task_ids"]) != all_task_ids for model in models[1:]):
        raise ValueError("evaluation runs must have the same expected_task_ids")
    success_sets = [set(model["success_ids"]) for model in models]
    solved_by = Counter(
        sum(task_id in success_ids for success_ids in success_sets)
        for task_id in all_task_ids
    )
    best = max(models, key=lambda model: model["success_rate"])
    return {
        "models": models,
        "outcome_order": [*REWARD_V3_TYPES, "unknown"],
        "histogram_labels": [
            f"{HISTOGRAM_EDGES[index]:.1f}～{min(HISTOGRAM_EDGES[index + 1], 1.0):.1f}"
            for index in range(len(HISTOGRAM_EDGES) - 1)
        ],
        "agreement": [{"models": count, "tasks": solved_by.get(count, 0)} for count in range(len(models) + 1)],
        "all_failed_task_ids": sorted(
            task_id
            for task_id in all_task_ids
            if not any(task_id in success_ids for success_ids in success_sets)
        ),
        "all_succeeded_task_ids": sorted(
            task_id
            for task_id in all_task_ids
            if all(task_id in success_ids for success_ids in success_sets)
        ),
        "best_model": best["label"],
    }


HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>多运行评测比较报告</title>
<style>
:root{--ink:#142033;--muted:#64748b;--line:#dbe4ef;--bg:#f3f6fa;--blue:#2563eb;--green:#16a34a;--amber:#d97706;--red:#dc2626;--violet:#7c3aed}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,-apple-system,"PingFang SC",sans-serif}
main{max-width:1440px;margin:auto;padding:28px}.hero,.card{background:#fff;border:1px solid var(--line);border-radius:16px;box-shadow:0 8px 28px #0f172a0a}
.hero{padding:28px;margin-bottom:18px;background:linear-gradient(135deg,#fff 55%,#e8f0ff)}h1{margin:0 0 8px;font-size:30px}h2{font-size:20px;margin:0 0 14px}h3{margin:0 0 8px}.muted{color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin:18px 0}.card{padding:20px;overflow:auto}.wide{grid-column:1/-1}
.kpis{display:grid;grid-template-columns:repeat(4,minmax(130px,1fr));gap:12px;margin-top:18px}.kpi{padding:14px;border-radius:12px;background:#f8fafc;border:1px solid var(--line)}.kpi b{display:block;font-size:24px}
table{border-collapse:collapse;width:100%;min-width:900px}th,td{padding:10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left;position:sticky;left:0;background:#fff}th{color:var(--muted);font-size:12px}
.bar-row{display:grid;grid-template-columns:150px 1fr 70px;gap:10px;align-items:center;margin:9px 0}.track{height:12px;background:#eef2f7;border-radius:99px;overflow:hidden}.fill{height:100%;background:var(--blue);border-radius:99px}.small{font-size:12px}.hist{display:grid;grid-template-columns:160px repeat(10,minmax(24px,1fr));gap:5px;align-items:end;margin:12px 0}.hist-name{align-self:center}.hist-bin{height:92px;background:#f1f5f9;display:flex;align-items:end;border-radius:5px 5px 0 0;overflow:hidden}.hist-bin i{display:block;width:100%;background:var(--violet)}.hist-labels{display:grid;grid-template-columns:160px repeat(10,minmax(24px,1fr));gap:5px;color:var(--muted);font-size:10px}.hist-labels span{writing-mode:vertical-rl;height:58px}
.stack{display:flex;height:18px;border-radius:99px;overflow:hidden;background:#eef2f7}.seg{height:100%}.outcome-row{display:grid;grid-template-columns:150px 1fr;gap:10px;align-items:center;margin:12px 0}.legend{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0}.legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px}
.notes{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.note{padding:14px;border:1px solid var(--line);border-radius:12px;background:#fbfdff}.note ul{padding-left:20px;margin:7px 0}.task-ids{word-break:break-all;padding:12px;background:#f8fafc;border-radius:10px;color:#475569}.links a{display:inline-block;margin:5px 10px 5px 0;padding:7px 11px;border-radius:9px;background:#eff6ff;color:#1d4ed8;text-decoration:none}
@media(max-width:900px){main{padding:14px}.grid{grid-template-columns:1fr}.kpis,.notes{grid-template-columns:1fr 1fr}.hist{grid-template-columns:100px repeat(10,minmax(15px,1fr))}.hist-labels{grid-template-columns:100px repeat(10,minmax(15px,1fr))}}
</style></head><body><main>
<section class="hero"><div class="muted">Shopping Agent</div><h1>多运行评测比较报告</h1><p>按目录 label 区分运行。严格成功率使用完整 benchmark 分母；Reward 统计仅包含已记录轨迹。</p><div class="kpis" id="kpis"></div><div class="links" id="links"></div></section>
<div class="grid">
<section class="card"><h2>严格成功率</h2><div id="success-bars"></div></section>
<section class="card"><h2>严格成功共识</h2><p class="muted">统计完整 expected_task_ids 集合中，每题被多少个运行记录为严格成功；缺失轨迹不算成功。</p><div id="agreement-bars"></div></section>
<section class="card wide"><h2>描述性统计</h2><p class="muted">成功率分母为 expected_tasks。Reward 分布与步数仅统计已记录轨迹；missing 不赋 Reward，空样本显示 —。</p><table><thead><tr><th>运行（目录 label）</th><th>已记录</th><th>缺失</th><th>严格成功</th><th>购买成功率</th><th>Reward 有效率</th><th>均值</th><th>中位数</th><th>标准差</th><th>最小</th><th>P25</th><th>P75</th><th>最大</th><th>平均步数</th></tr></thead><tbody id="stats"></tbody></table></section>
<section class="card wide"><h2>Reward 分布（仅已记录轨迹）</h2><p class="muted">每一小柱是一个 0.2 宽区间；紫柱越高，落在该 Reward 区间的已记录任务越多。缺失轨迹不计入。</p><div id="histograms"></div><div class="hist-labels" id="hist-labels"></div></section>
<section class="card wide"><h2>终局类型分布（仅已记录轨迹）</h2><p class="muted">各类型宽度以完整 benchmark 为分母；未填满部分为缺失轨迹，不是 Reward 类型。</p><div class="legend" id="legend"></div><div id="outcomes"></div></section>
<section class="card wide"><h2>本批数据事实</h2><div id="best-success"></div><div class="notes" id="run-facts"></div></section>
<section class="card wide"><h2>严格成功任务集合</h2><h3>所有运行均未记录严格成功的任务（<span id="all-failed-count"></span>）</h3><p class="muted">该集合包含缺失轨迹，不能全部归因于任务失败。</p><div class="task-ids" id="all-failed"></div><h3 style="margin-top:16px">所有运行均严格成功的任务（<span id="all-succeeded-count"></span>）</h3><div class="task-ids" id="all-succeeded"></div></section>
</div></main>
<script>
const D=__REPORT_DATA__;
const COLORS={gold_purchase:'#16a34a',valid_alternative_purchase:'#4ade80',partial_alternative_purchase:'#f59e0b',wrong_purchase:'#ef4444',repeat_loop:'#7c3aed',max_steps:'#db2777',early_abstain:'#94a3b8',graceful_stop:'#38bdf8',reward_unverifiable:'#a16207',unknown:'#cbd5e1'};
const LABELS={gold_purchase:'正确购买',valid_alternative_purchase:'有效替代品',partial_alternative_purchase:'部分匹配',wrong_purchase:'买错商品',repeat_loop:'重复循环',max_steps:'步数耗尽',early_abstain:'过早放弃',graceful_stop:'主动停止',reward_unverifiable:'无法核验',unknown:'已记录 / 无已知终局类型'};
const pct=x=>(x*100).toFixed(1)+'%'; const n=(x,d=3)=>x===null?'—':Number(x).toFixed(d); const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const sorted=[...D.models].sort((a,b)=>b.success_rate-a.success_rate); const total=D.models[0].tasks;
document.querySelector('#kpis').innerHTML=`<div class="kpi"><span>运行</span><b>${D.models.length}</b></div><div class="kpi"><span>每运行 benchmark 任务</span><b>${total}</b></div><div class="kpi"><span>最高严格成功率</span><b>${pct(sorted[0].success_rate)}</b><small>${esc(sorted[0].label)}</small></div><div class="kpi"><span>全运行未记录严格成功</span><b>${D.all_failed_task_ids.length}</b></div>`;
document.querySelector('#links').innerHTML=sorted.map(m=>`<a href="${encodeURI(m.report)}">${esc(m.label)} 单运行报告</a>`).join('');
function bars(target,items,max,label){document.querySelector(target).innerHTML=items.map(x=>`<div class="bar-row"><span>${esc(x.name)}</span><div class="track"><div class="fill" style="width:${max>0?x.value/max*100:0}%"></div></div><b>${label(x)}</b></div>`).join('')}
bars('#success-bars',sorted.map(m=>({name:m.label,value:m.success_rate,count:m.successes})),1,x=>`${x.count}/${total}`);
bars('#agreement-bars',D.agreement.map(x=>({name:`${x.models} 个模型`,value:x.tasks})),Math.max(...D.agreement.map(x=>x.tasks)),x=>x.value);
document.querySelector('#stats').innerHTML=sorted.map(m=>`<tr><td><a href="${encodeURI(m.report)}">${esc(m.label)}</a></td><td>${m.recorded}</td><td>${m.missing}</td><td>${m.successes}/${m.tasks}（${pct(m.success_rate)}）</td><td>${pct(m.purchase_rate)}</td><td>${pct(m.reward_valid_rate)}</td><td>${n(m.reward.mean)}</td><td>${n(m.reward.median)}</td><td>${n(m.reward.stddev)}</td><td>${n(m.reward.min)}</td><td>${n(m.reward.q1)}</td><td>${n(m.reward.q3)}</td><td>${n(m.reward.max)}</td><td>${n(m.average_steps,2)}</td></tr>`).join('');
document.querySelector('#histograms').innerHTML=sorted.map(m=>{const max=Math.max(...m.reward.histogram);return `<div class="hist"><b class="hist-name">${esc(m.label)}（n=${m.reward.count}）</b>${m.reward.histogram.map(v=>`<div class="hist-bin" title="${v} 条"><i style="height:${max>0?v/max*100:0}%"></i></div>`).join('')}</div>`}).join('');
document.querySelector('#hist-labels').innerHTML='<b></b>'+D.histogram_labels.map(x=>`<span>${x}</span>`).join('');
document.querySelector('#legend').innerHTML=D.outcome_order.map(k=>`<span><i style="background:${COLORS[k]}"></i>${LABELS[k]}</span>`).join('');
document.querySelector('#outcomes').innerHTML=sorted.map(m=>`<div class="outcome-row"><b>${esc(m.label)}<br><small>已记录 ${m.recorded} / 缺失 ${m.missing}</small></b><div class="stack">${D.outcome_order.map(k=>`<div class="seg" title="${LABELS[k]}：${m.outcomes[k]||0}" style="width:${m.tasks?(m.outcomes[k]||0)/m.tasks*100:0}%;background:${COLORS[k]}"></div>`).join('')}</div></div>`).join('');
const best=sorted[0];
document.querySelector('#best-success').innerHTML=`<p>本批最高严格成功率：<b>${esc(best.label)}</b>，${best.successes}/${best.tasks}（${pct(best.success_rate)}）。</p>`;
const countsText=counts=>Object.entries(counts).map(([k,v])=>`${esc(k)}：${v}`).join('；')||'无记录';
document.querySelector('#run-facts').innerHTML=sorted.map(m=>`<article class="note"><h3>${esc(m.label)}</h3><p>模型：${esc(m.name)}</p><p>已记录 ${m.recorded}/${m.tasks}；missing ${m.missing}（不是 Reward）。</p><p>终局：${countsText(m.outcomes)}</p><p>Status：${countsText(m.statuses)}</p><p>Guards（拒绝次数）：${countsText(m.guards)}</p><p>缺失 task IDs：${m.missing_task_ids.join(', ')||'无'}</p></article>`).join('');
document.querySelector('#all-failed-count').textContent=D.all_failed_task_ids.length;document.querySelector('#all-failed').textContent=D.all_failed_task_ids.join(', ');document.querySelector('#all-succeeded-count').textContent=D.all_succeeded_task_ids.length;document.querySelector('#all-succeeded').textContent=D.all_succeeded_task_ids.join(', ');
</script></body></html>'''


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="生成多运行评测比较 HTML 报告")
    parser.add_argument(
        "--evaluation-dir", type=Path, default=ROOT / "outputs" / "evaluation"
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = args.output or args.evaluation_dir / "comparison-report.html"
    data = json.dumps(build_comparison_data(args.evaluation_dir), ensure_ascii=False, separators=(",", ":"))
    output.write_text(HTML.replace("__REPORT_DATA__", data.replace("</", "<\\/")), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
