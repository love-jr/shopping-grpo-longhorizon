# Final-200 Clean 评估

## 运行

保持 ShopSimulator 和 Actor 服务运行。默认只执行 rollout、确定性指标计算和 HTML 报告，
不生成 Rubric、不调用 Flash Curator/Judge，也不要求 Flash API 凭据：

```bash
bash scripts/evaluate.sh sft
```

`LLM_BASE_URL / LLM_API_KEY / SERVED_MODEL_NAME` 只配置 Actor，默认指向本地 vLLM。
只有明确需要额外语义诊断时，才显式开启 Rubric 与轨迹 Judge：

```bash
export OPENAI_BASE_URL=https://your-provider.example/v1
export OPENAI_API_KEY=your-key
EVAL_RUBRIC_JUDGE=1 bash scripts/evaluate.sh sft
```

`OPENAI_BASE_URL / OPENAI_API_KEY` 仅在此模式下用于 `deepseek-v4.1-flash`。
启用后追加 TaskFacts → 代码候选 → Flash Rubric → 轨迹预处理 → Judge 输入 →
Flash Judge → 四面板汇总；严格成功率和 Reward 始终由代码计算，不依赖 Judge。

## 固定 Benchmark

正式分母为 [`data/evaluation/tasks.jsonl`](../data/evaluation/tasks.jsonl) 的 200 个 task_id。
SHA-256：`d99112a20ef47534c27a32e4b38229bf048dcc6b06fef2e3e919aac3093662f5`。
环境契约：ShopSimulator v2.1、Reward v3、observation/tool schema v2。

Final-200 Clean 替换了 81 道不可严格评分或 Query/Gold 冲突的题；筛选要求 gold 购买可达、
规格无歧义、variant 价格可解、Query 价格实际参与预算解析，且与训练/验证 task ID 零重叠。
剔除、补入清单和固定抽样 seed 存于 [`metadata.json`](../data/evaluation/metadata.json)，不在文档复制。
这不证明所有替代商品都可正确评分。

每题一次确定性 rollout：temperature/top-p `0/1`、最多 35 环境步、每回合 512 tokens、
context 24,576，关闭 compaction。缺失、基础设施失败和 `not_judged` 都保留在 200 题分母。
严格成功要求完整 `gold_purchase` 终局且 `reward_valid=true`。
不使用盲测集调 Prompt、校准 Judge 或选 checkpoint。

## Rubric 与 Judge 契约

- 代码拥有候选的 `field_path / operator / expected_value`；Flash 只能选择 candidate、描述和 hard/soft。
- 每条选择必须提供非空、逐字存在于 Query 的 quote；空 Rubric 或空证据均拒绝，不回退到候选 spans。
- 同一 Benchmark 的 Baseline、SFT、GRPO 共用冻结 Rubric。
- Judge 只看 Query、Rubric、Actor 可见轨迹、中性终局标志和白名单指标；不接收 Reward、Gold 私有字段或 raw observation。
- Judge 逐条判断 Rubric，并对搜索、候选利用、证据核验、决策、终止效率分别给出 0/1/2 分，引用真实 event_id。
- 输出按固定契约校验；无效模型结果不写入缓存，不进行 schema repair 或默认评分。

完整 Prompt 只维护在 [`prompts.py`](../src/shopping_grpo/evaluation/prompts.py)：
`rubric-curator-v1-draft-r4` / `trajectory-judge-v1-draft-r4`。
结果分为 Reward/终局、Query Rubric、轨迹质量、确定性行为/基础设施四个面板；不合成总分。

## 产物与续跑

```text
outputs/evaluation/judge-shared/{task_facts,rubric_candidates,rubrics}.jsonl
outputs/evaluation/NAME/{trajectories.jsonl,summary.json,report.html}
outputs/evaluation/NAME/judge/{preprocessed,judge_requests,judges,evaluations}.jsonl
outputs/evaluation/NAME/judge/evaluation_summary.json
```

默认仅生成 `trajectories.jsonl`、`summary.json`、`report.html`；共享资产和 `judge/`
仅在 `EVAL_RUBRIC_JUDGE=1` 时生成或复用。`EVAL_RESUME / EVAL_FORCE` 不会隐式开启评审。

共享 Rubric 每次校验缓存并只补缺失任务，不重复调用已完成的 Curator。
`EVAL_RESUME=1` 补缺失 Judge 结果；输入变更则拒绝旧缓存。
`EVAL_FORCE=1` 重跑当前模型 Judge，不覆盖跨 Actor 的冻结 Rubric。
修改数据、extractor 或 Prompt 后使用新 `EVAL_SHARED_DIR` 并重新评估所有 Actor，不能混用标准。

单独运行 `scripts/eval_rubric_judge.py` 默认拒绝盲测 task ID；正式处理须显式传 `--allow-blind-final`。
阶段参数见 `--help`；配对比较用 `compare --model-evaluations LABEL=PATH ...`。
产物不提交 Git。修改 Benchmark 后同步 metadata 与 packaged blind guard，并运行
`tests/test_evaluation_dataset.py`；重算历史分母不等于重新 rollout。
