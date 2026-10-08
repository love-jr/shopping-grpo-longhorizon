import unittest
from types import SimpleNamespace
import json
import tempfile

from unittest.mock import patch
import pyarrow.parquet
import torch
from tensordict import TensorDict
from verl.workers.rollout.vllm_rollout.utils import extract_prompt_logprobs
from verl.workers.utils.padding import no_padding_2_padding

from shopping_grpo.training.opsd.privilege import purchase_contract, reference_rejection_reasons, teacher_messages
from shopping_grpo.training.opsd.worker import ShoppingOPSDWorker, align_teacher_response, single_teacher

import sys
from pathlib import Path
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_grpo_runtime  # noqa: E402


def valid_contract():
    return purchase_contract(
        {"instruction_text": "镜子，预算四百元", "attributes": [], "goal_options": [],
         "unresolved_option_requirements": [], "price_upper": 400.0},
        {"Title": "镜子", "category": "镜子"},
        {"selected_options": {}, "variant_price": 300.0},
    )


class OPSDPreflightTest(unittest.TestCase):
    """Preflight requires usable references for the exact training task set."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _train_parquet(self, task_ids):
        path = self.root / "train.parquet"
        pyarrow.parquet.write_table(
            pyarrow.Table.from_pylist(
                [{"extra_info": {"split": "train", "index": i, "task_id": t}}
                 for i, t in enumerate(task_ids)],
                schema=pyarrow.schema([("extra_info", pyarrow.struct([
                    ("split", pyarrow.string()), ("index", pyarrow.int64()),
                    ("task_id", pyarrow.int64()),
                ]))]),
            ),
            path,
        )
        return path

    def _config(self, teacher_models, contract_ids, train_ids=(5,), **loss):
        path = self.root / "contracts.json"
        path.write_text(
            json.dumps({"product_data_sha256": "x", "contracts": {str(i): valid_contract() for i in contract_ids}}),
            encoding="utf-8",
        )
        return OmegaConf.create(
            {
                "distillation": {
                    "enabled": True,
                    "distillation_loss": {"use_task_rewards": False, **loss},
                    "teacher_models": teacher_models,
                },
                "shopping_opsd": {"contracts_path": str(path)},
                "data": {"train_files": str(self._train_parquet(train_ids))},
            }
        )

    def test_accepts_contracts_covering_every_training_task(self):
        check_grpo_runtime.validate_opsd(self._config({"teacher_model": {}}, [5]))

    def test_rejects_contracts_missing_a_training_task(self):
        with self.assertRaises(SystemExit):
            check_grpo_runtime.validate_opsd(self._config({"teacher_model": {}}, [6]))

    def test_rejects_training_tasks_that_leak_the_evaluation_set(self):
        # 8187 是冻结评测集任务；make it a training task to trip the disjointness guard.
        with self.assertRaises(SystemExit):
            check_grpo_runtime.validate_opsd(
                self._config({"teacher_model": {}}, [5, 8187], train_ids=(5, 8187))
            )

    def test_rejects_task_rewards_next_to_distillation(self):
        with self.assertRaises(SystemExit):
            check_grpo_runtime.validate_opsd(
                self._config({"teacher_model": {}}, [5], use_task_rewards=True)
            )

    def test_rejects_excess_contracts(self):
        with self.assertRaisesRegex(SystemExit, "match exactly"):
            check_grpo_runtime.validate_opsd(self._config({"teacher_model": {}}, [5, 6]))

    def test_rejects_empty_training_set(self):
        with self.assertRaisesRegex(SystemExit, "must not be empty"):
            check_grpo_runtime.validate_opsd(self._config({"teacher_model": {}}, [], train_ids=()))

    def test_rejects_missing_request_or_product_title(self):
        for field in ("user_request", "title"):
            for value in (None, "", " \n ", 123):
                with self.subTest(field=field, value=value):
                    config = self._config({"teacher_model": {}}, [5])
                    contract = valid_contract()
                    target = contract if field == "user_request" else contract["reference_purchase"]
                    target[field] = value
                    self.assertTrue(reference_rejection_reasons(contract))
                    Path(config.shopping_opsd.contracts_path).write_text(json.dumps({"contracts": {"5": contract}}))
                    with self.assertRaisesRegex(SystemExit, "unusable reference"):
                        check_grpo_runtime.validate_opsd(config)
        self.assertTrue(reference_rejection_reasons({}))
        self.assertTrue(reference_rejection_reasons(None))

    def test_accepts_unknown_budget_price_and_unresolved_options(self):
        config = self._config({"teacher_model": {}}, [5])
        contract = valid_contract()
        contract["annotations"]["parsed_budget_upper"] = None
        contract["annotations"]["unresolved_options"] = [{"value": "银色", "reason": "axis_not_found"}]
        contract["reference_purchase"]["variant_price"] = None
        self.assertEqual(reference_rejection_reasons(contract), [])
        Path(config.shopping_opsd.contracts_path).write_text(json.dumps({"contracts": {"5": contract}}))
        check_grpo_runtime.validate_opsd(config)

    def test_accepts_product_reference_without_options_or_price_fields(self):
        contract = {"user_request": "买一面镜子", "reference_purchase": {"title": "镜子"}}
        self.assertEqual(reference_rejection_reasons(contract), [])

    def test_price_does_not_filter_reference(self):
        contract = valid_contract()
        contract["reference_purchase"]["variant_price"] = 401.0
        self.assertEqual(reference_rejection_reasons(contract), [])

    def test_worker_rejects_missing_title_when_preflight_is_bypassed(self):
        config = self._config({"teacher_model": {}}, [5])
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"product_data_sha256": "x"}))
        config.shopping_opsd.manifest_path = str(manifest)
        contract = valid_contract()
        contract["reference_purchase"].pop("title")
        Path(config.shopping_opsd.contracts_path).write_text(json.dumps({
            "product_data_sha256": "x", "contracts": {"5": contract},
        }))

        def initialize(worker, *args, **kwargs):
            worker.config = config

        with patch("shopping_grpo.training.opsd.worker.AgentLoopWorker.__init__", initialize):
            with self.assertRaisesRegex(ValueError, "title"):
                ShoppingOPSDWorker()

class OPSDTeacherRoutingTest(unittest.TestCase):
    """One contract is appended to every teacher system prompt, so routing cannot work."""

    def test_single_teacher_returns_it_and_rejects_a_second(self):
        config = OmegaConf.create({"distillation": {"teacher_models": {"a": {"k": 1}}}})
        self.assertEqual(single_teacher(config)["k"], 1)
        two = OmegaConf.create({"distillation": {"teacher_models": {"a": {}, "b": {}}}})
        with self.assertRaises(ValueError):
            single_teacher(two)


class OPSDCoreTest(unittest.TestCase):
    def test_contract_preserves_uncertain_annotations_and_product_reference(self):
        goal = {
            "instruction_text": "银色八英寸，预算三百九十元内",
            "attributes": ["三色光"],
            "goal_options": ["银色八英寸?三色光"],
            "unresolved_option_requirements": [
                {"value": "银色八英寸?三色光", "reason": "axis_not_found"}
            ],
            "price_upper": None,
            "asin": "hidden",
            "reward": 1.0,
            "user_persona": {"hidden": True},
        }
        reference = {
            "selected_options": {"尺寸": "8英寸"},
            "variant_price": 400.0,
            "asin": "hidden-reference-id",
            "reward": 1.0,
        }
        contract = purchase_contract(
            goal,
            {"Title": "化妆镜", "shop_name": "镜子专营店", "category": "镜子"},
            reference,
        )
        self.assertEqual(contract["user_request"], goal["instruction_text"])
        self.assertEqual(contract["annotations"]["option_values"], goal["goal_options"])
        self.assertEqual(contract["annotations"]["unresolved_options"], goal["unresolved_option_requirements"])
        self.assertIsNone(contract["annotations"]["parsed_budget_upper"])
        answer = contract["reference_purchase"]
        self.assertEqual(answer["shop_name"], "镜子专营店")
        self.assertNotIn("brand", answer)
        self.assertEqual(answer["selected_options"], {"尺寸": "8英寸"})
        self.assertEqual(answer["variant_price"], 400.0)
        self.assertEqual(answer["available_options"], {})
        self.assertNotIn("verification", answer)
        self.assertNotIn("inferred_options", answer)
        encoded = str(contract)
        self.assertNotIn("reward", encoded)
        self.assertNotIn("persona", encoded)
        self.assertNotIn("asin", encoded)

    def test_teacher_context_does_not_mutate_student_messages(self):
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
        teacher = teacher_messages(messages, {"reference_purchase": {"title": "化妆镜"}})
        self.assertEqual(messages, [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}])
        self.assertEqual(teacher[1], messages[1])
        self.assertIn("化妆镜", teacher[0]["content"])

    def test_alignment_discards_teacher_prefix(self):
        # Teacher prompt=3, student prompt=2, response=2.  The real vLLM
        # extractor left-shifts next-token scores; native padding shifts again.
        result = {}
        extract_prompt_logprobs(
            SimpleNamespace(prompt_logprobs=[None] + [
                {token: SimpleNamespace(logprob=-float(token), rank=1)}
                for token in (11, 12, 21, 22)
            ]),
            num_prompt_logprobs=1,
            result_dict=result,
        )
        ids, logprobs = align_teacher_response(
            torch.tensor(result["prompt_ids"]), torch.tensor(result["prompt_logprobs"]),
            prompt_length=2, teacher_prompt_length=3,
        )
        data = TensorDict({
            "prompts": torch.tensor([[1, 2]]),
            "responses": torch.tensor([[21, 22]]),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        }, batch_size=[1])
        self.assertEqual([21, 22], no_padding_2_padding(ids, data).flatten().tolist())
        self.assertEqual([-21.0, -22.0], no_padding_2_padding(logprobs, data).flatten().tolist())


if __name__ == "__main__":
    unittest.main()
