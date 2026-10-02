# 数据采集、SFT 与 GRPO

正式路径：Baseline → SFT → GRPO → Evaluation。安装和服务启动见
[README](../README.md#快速开始)；只维护这一条默认训练路径。

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

配置以 [`configs/grpo.yaml`](../configs/grpo.yaml) 为准：每题 4 条 rollout，
学习率 `1e-6`，最多 500 optimizer steps；每 50 步保存和验证。
动态采样最多重采 3 批，最多连续跳过 10 次无信号更新。

双卡配置需要 remove-padding、Triton fused kernels 和
`RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`；小 `/dev/shm` 容器使用
`data.dataloader_num_workers=0`。诊断写入运行目录的 `training_diagnostics.jsonl`。

导出后按[评估指南](evaluation.md)评估；checkpoint 选择只使用验证集，不使用 Final-200 Clean。
