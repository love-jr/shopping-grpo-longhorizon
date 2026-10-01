# Reward v3

环境直接计算终局 Reward，不使用 LLM Judge。任务特征在 rollout 前冻结。
类别与 variant 预算为 hard gate；未通过得 `-0.85`，无法核验得 `0` 且 `reward_valid=false`。

品牌、型号、核心功能、规格权重为 `0.35 / 0.25 / 0.25 / 0.15`，仅在活跃维度归一化：
`S = Σ(weight × score) / Σ(active weight)`。完全满足要求 match 和 evidence coverage 都为 1。

| 终局 | Reward |
|---|---:|
| 完全满足的目标 ASIN：`gold_purchase` | 1.00 |
| 完全满足的替代商品 | 0.55 |
| 部分满足的替代商品 | `min(0.25, -0.30 + 0.55 × S)` |
| 充分搜索后合理放弃 / 过早放弃 | -0.15 / -0.35 |
| 最大 35 步 / 重复无进展 | -0.50 / -0.65 |
| 类别错误或超预算 | -0.85 |
| 证据不可核验 | 0，invalid |

合理放弃要求至少两个有效结果集、两个候选，且没有已知可接受候选；可接受候选要求
hard gate 通过、match ≥ 0.70、coverage ≥ 0.75。
环境在连续两次完全重复、连续四次无新证据或 35 步时终止。
严格成功还要求完整 `gold_purchase` 终局和 `reward_valid=true`，不能仅看 reward 数值。

实现与常量是唯一真源：
[`environment.json`](../environments/ShopSimulator/shop_env/configs/environment.json)、
[`reward.py`](../environments/ShopSimulator/shop_env/web_agent_site/engine/reward.py)、
[`termination.py`](../environments/ShopSimulator/shop_env/web_agent_site/engine/termination.py)。
