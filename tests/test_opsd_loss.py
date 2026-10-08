import unittest

import torch
from tensordict import TensorDict
from verl.trainer.distillation.losses import distillation_ppo_loss as native_loss
from verl.utils.tensordict_utils import assign_non_tensor
from verl.workers.config import ActorConfig, DistillationConfig, DistillationLossConfig

from shopping_grpo.training.opsd.loss import distillation_ppo_loss


class OPSDLossTest(unittest.TestCase):
    def _configs(self):
        actor = ActorConfig(
            strategy="fsdp",
            rollout_n=1,
            use_dynamic_bsz=True,
            loss_agg_mode="token-mean",
        )
        distillation = DistillationConfig(
            distillation_loss=DistillationLossConfig(
                loss_mode="k1",
                use_task_rewards=False,
                use_policy_gradient=True,
                loss_max_clamp=5.0,
                log_prob_min_clamp=None,
            )
        )
        return actor, distillation

    def _micro_batch(self, lengths, parameter, total_tokens, total_sequences):
        width = max(lengths)
        mask = torch.arange(width).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
        # One prompt token per sequence. Native padding left-shifts packed
        # next-token scores, so each sequence has length+1 packed scores.
        packed_length = sum(length + 1 for length in lengths)
        log_probs = parameter.expand(packed_length) - 1.0
        data = TensorDict(
            {
                "prompts": torch.ones(len(lengths), 1, dtype=torch.long),
                "responses": torch.ones(len(lengths), width, dtype=torch.long),
                "attention_mask": torch.cat(
                    (torch.ones(len(lengths), 1, dtype=torch.bool), mask), dim=1
                ),
                "response_mask": mask,
                "old_log_probs": torch.full((len(lengths), width), -1.0, dtype=torch.float64),
                "advantages": torch.zeros(len(lengths), width, dtype=torch.float64),
            },
            batch_size=[len(lengths)],
        )
        # Packed teacher scores need not share the TensorDict batch dimension.
        data.set_non_tensor("teacher_logprobs", torch.full((packed_length, 1), -1.5, dtype=torch.float64))
        assign_non_tensor(
            data,
            dp_size=1,
            batch_num_tokens=total_tokens,
            global_batch_size=total_sequences,
        )
        return {"log_probs": log_probs}, data

    def _evaluate(self, loss_fn, actor, distillation, lengths, partitions):
        parameter = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        total_loss = torch.zeros((), dtype=torch.float64)
        start = 0
        self.assertEqual(sum(partitions), len(lengths))
        for size in partitions:
            model_output, data = self._micro_batch(
                lengths[start : start + size],
                parameter,
                total_tokens=sum(lengths),
                total_sequences=len(lengths),
            )
            loss, _ = loss_fn(
                config=actor,
                distillation_config=distillation,
                model_output=model_output,
                data=data,
            )
            loss.backward()
            total_loss += loss.detach()
            start += size
        return total_loss.item(), parameter.grad.item()

    def test_unequal_minis_and_micro_batches_match_unsplit_loss_and_gradient(self):
        for lengths in ((1, 2, 1), (3, 1, 2)):
            for partitions in ((3,), (1, 2), (2, 1), (1, 1, 1)):
                with self.subTest(lengths=lengths, partitions=partitions):
                    actor, distillation = self._configs()
                    actual = self._evaluate(distillation_ppo_loss, actor, distillation, lengths, partitions)
                    reference = self._evaluate(distillation_ppo_loss, *self._configs(), lengths, (3,))
                    for value, expected in zip(actual, reference, strict=True):
                        self.assertAlmostEqual(value, expected, places=12)
                        self.assertAlmostEqual(value, 0.5, places=12)

    def test_previous_mini_token_counts_do_not_change_current_loss_or_gradient(self):
        for previous_lengths in ((1, 1, 1), (1, 2, 1), (3, 3, 3)):
            with self.subTest(previous_lengths=previous_lengths):
                actor, distillation = self._configs()
                self._evaluate(distillation_ppo_loss, actor, distillation, previous_lengths, (1, 2))
                actual = self._evaluate(distillation_ppo_loss, actor, distillation, (3, 1, 2), (1, 2))
                self.assertAlmostEqual(actual[0], 0.5, places=12)
                self.assertAlmostEqual(actual[1], 0.5, places=12)

    def test_native_loss_exposes_first_call_and_stale_mini_denominators(self):
        # Failing-before witness using the unmodified native numerical loss.
        # Current mini has six tokens: each micro contributes 1.5 to the sum.
        actor, distillation = self._configs()
        first_call = self._evaluate(native_loss, actor, distillation, (3, 1, 2), (1, 2))
        self.assertAlmostEqual(first_call[0], 0.75, places=12)
        self.assertAlmostEqual(first_call[1], 0.75, places=12)
        for previous_lengths, expected in (((1, 1, 1), 0.75), ((1, 2, 1), 0.625), ((3, 3, 3), 5 / 12)):
            with self.subTest(previous_lengths=previous_lengths):
                actor, distillation = self._configs()
                self._evaluate(native_loss, actor, distillation, previous_lengths, (3,))
                stale = self._evaluate(native_loss, actor, distillation, (3, 1, 2), (1, 2))
                self.assertAlmostEqual(stale[0], expected, places=12)
                self.assertAlmostEqual(stale[1], expected, places=12)
                self.assertNotAlmostEqual(stale[0], 0.5, places=12)
                self.assertNotAlmostEqual(stale[1], 0.5, places=12)


if __name__ == "__main__":
    unittest.main()
