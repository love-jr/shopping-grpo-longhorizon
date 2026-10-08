# 数据采集、SFT、GRPO 与 OPSD

项目流程：Baseline → SFT → GRPO → OPSD → Evaluation。GRPO 与 OPSD 分别从
SFT 权重训练并统一评测。安装和服务启动见[README](../README.md#快速开始)。

## 数据采集

现有教师为 `deepseek-v4-flash`：2,498 条原始轨迹中，1,026 条通过完整
`gold_purchase` 且 `reward_valid=true` 的筛选；冻结使用 800 train / 200 validation。
数据来源、哈希和审计见 [`data/sft/metadata.json`](../data/sft/metadata.json)。
训练、验证与 [`data/evaluation/tasks.jsonl`](../data/evaluation/tasks.jsonl) 必须 task-disjoint。

```bash
export OPENAI_BASE_URL=https://your-provider.example/v1
export OPENAI_API_KEY=your-key
.venv/bin/python scripts/collect_sft_data.py \
  --tasks data/grpo/train.jsonl --output-dir outputs/sft-collection \
  --model deepseek-v4-flash --target-accepted 1000 --workers 4
```

重复运行会从 `raw.jsonl` 续跑；只重建派生数据时加 `--build-only`。
审核 `reject_stats.json`、数据划分和 metadata 后再更新 `data/sft/`；原始轨迹不提交。

教师采集默认保留完整 observation，不依赖 vLLM 的 `/tokenize`；GRPO/本地评估按
[评估指南](evaluation.md)做 token 预算投影。需要完全一致的输入时，采集服务必须支持
`/tokenize`，并显式传 `--observation-token-budget 4096`；已有冻结数据不自动重写。

## SFT

```bash
bash scripts/sft.sh
```

- 基础模型：`Qwen/Qwen3.5-2B`；输入：`data/sft/{train,validation}.jsonl`。
- 仅 assistant 动作 token 计算 loss，user 和 tool token 被 mask。
- 默认 3 epoch、24,576 context、batch 1、gradient accumulation 8、学习率 `1e-4`。
- LoRA rank/alpha 为 16/32；启用 gradient checkpointing、SDPA 与 Liger。
- 输出：`outputs/models/sft-lora` 和合并后的 `outputs/models/sft-merged`。

参数以 [`scripts/train_lora_sft.py`](../scripts/train_lora_sft.py) 为准。
训练前停止 Actor 模型服务，避免显存竞争。

## GRPO

输入为 `outputs/models/sft-merged`，训练/验证集为
`data/grpo/{train,validation}.parquet`（1,000 / 50 tasks）。
使用固定 `verl==0.8.0`、项目 AgentLoop/工具适配层及 setup 中的 SHA-256 校验补丁；
不复制 veRL 源码。Reward 直接来自环境，不使用 LLM Judge 训练奖励。
工具 Schema 只维护在 `src/shopping_grpo/environment/tools.py`。
GRPO 启动时生成运行目录的 `tools.json`，训练读取该文件；`--dry-run` 不写产物。

```bash
bash scripts/grpo.sh --dry-run
bash scripts/grpo.sh
# 按验证集选择 checkpoint，再导出
bash scripts/export_grpo.sh \
  outputs/models/grpo/global_step_100/actor outputs/models/grpo-merged
```

导出产物的 LoRA 在 `lora_adapter/`，不在主体 `model.safetensors` 中。
`serve_model.sh` 会加载 LoRA 并将 `shopping-agent` 指向它；`shopping-agent-base` 是未应用 LoRA 的基座。
启动后从 `/v1/models` 核对 adapter，使用独立标签保存各 checkpoint 的评估结果。

配置以 [`configs/grpo.yaml`](../configs/grpo.yaml) 为准：每题 4 条 rollout，
学习率 `1e-6`，最多 500 optimizer steps；每 50 步保存和验证。
动态采样最多重采 3 批，最多连续跳过 10 次无信号更新。

双卡配置需要 remove-padding、Triton fused kernels 和
`RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`；小 `/dev/shm` 容器使用
`data.dataloader_num_workers=0`。诊断写入运行目录的 `training_diagnostics.jsonl`。

## OPSD

学生自行 rollout；冻结教师从同一份 SFT 权重加载，额外读取私有商品参考，
对学生动作 token 提供 k1 蒸馏信号。`use_task_rewards=false`，不优化购物奖励。

`prepare_opsd.py` 为全部 1,000 个训练任务提取原始需求、规格标注与目标商品信息，
不提供 ASIN、persona 或奖励校验结果。仅匹配明确规格，不补选最低价选项；
没有规格无需构造变体，未知价格留空，不排除任务。标注不能覆盖用户原话，
教师只能操作当前页面可见目标。学生、验证和评测均看不到参考。

actor、teacher 各用一张 GPU；启动前停止评测服务，确保 `/dev/shm` 空间充足。
训练直接使用原始 parquet，preflight 检查参考覆盖、评测集零重叠与必要字段。
配置见 [`configs/opsd.yaml`](../configs/opsd.yaml)：

```bash
bash scripts/opsd.sh -- \
  data.train_batch_size=8 trainer.total_training_steps=200 trainer.save_freq=50
```

输出目录必须新建或为空，可用 `OPSD_OUTPUT_DIR` 指定。每 50 步验证和保存，
checkpoint 只按验证集选择。导出与服务启动复用 `export_grpo.sh`、`serve_model.sh`，
评测见[评估指南](evaluation.md)。当前版本 step 200 最新完整评测为 **138/200（69.0%）**；
验证 reward 与严格成功率不是同一指标。

