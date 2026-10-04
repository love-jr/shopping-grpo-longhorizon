import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_comparison_report import build_comparison_data
from shopping_grpo.evaluation.summary import REWARD_V3_TYPES, summarize_trajectories


class ComparisonReportTest(unittest.TestCase):
    def _write_run(self, root, label, expected_ids, rows):
        run = root / label
        run.mkdir()
        summary = summarize_trajectories(expected_ids, rows)
        summary["protocol"] = {"model": "shopping-agent"}
        (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (run / "trajectories.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return run

    def _success(self, task_id):
        return {
            "task_id": task_id,
            "status": "done",
            "done": True,
            "final_reward": 1.0,
            "terminal_result": {
                "done": True,
                "over": True,
                "reward_detail": {
                    "reward_version": "shopsimulator-reward-v3",
                    "reward_type": "gold_purchase",
                    "reward_valid": True,
                    "purchase_success": True,
                    "termination_reason": "gold_purchase",
                },
            },
        }

    def test_discovers_runs_and_computes_reward_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [self._success(1), {"task_id": 2, "final_reward": -0.5}]
            self._write_run(root, "model-a", [1, 2], rows)

            data = build_comparison_data(root)

            self.assertEqual([model["key"] for model in data["models"]], ["model-a"])
            model = data["models"][0]
            self.assertEqual(model["name"], "shopping-agent")
            self.assertEqual(model["label"], "model-a")
            self.assertEqual(model["success_ids"], [1])
            self.assertEqual(model["reward"]["mean"], 0.25)
            self.assertEqual(model["reward"]["median"], 0.25)
            self.assertEqual(sum(model["reward"]["histogram"]), 2)

    def test_empty_strict_success_ids_do_not_fall_back_to_gold_purchase(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self._success(1)
            row["terminal_result"]["reward_detail"]["reward_valid"] = False
            self._write_run(root, "baseline", [1], [row])

            data = build_comparison_data(root)

            model = data["models"][0]
            self.assertEqual(model["success_ids"], [])
            self.assertEqual(model["successes"], 0)
            self.assertEqual(model["success_rate"], 0.0)
            self.assertEqual(model["outcomes"], {"gold_purchase": 1})
            self.assertEqual(data["agreement"], [{"models": 0, "tasks": 1}, {"models": 1, "tasks": 0}])

    def test_missing_tasks_keep_fixed_denominator_and_expected_agreement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected_ids = list(range(1, 201))
            self._write_run(root, "baseline", expected_ids, [self._success(1)])
            self._write_run(root, "sft", expected_ids, [{"task_id": 2, "final_reward": -0.5}])
            self._write_run(root, "grpo", expected_ids, [])

            data = build_comparison_data(root)

            models = {model["label"]: model for model in data["models"]}
            self.assertEqual(set(models), {"baseline", "sft", "grpo"})
            self.assertTrue(all(model["tasks"] == 200 for model in models.values()))
            self.assertEqual(models["baseline"]["recorded"], 1)
            self.assertEqual(models["baseline"]["missing"], 199)
            self.assertEqual(models["baseline"]["success_rate"], 1 / 200)
            self.assertEqual(models["baseline"]["reward"]["mean"], 1.0)
            self.assertEqual(models["grpo"]["recorded"], 0)
            self.assertEqual(models["grpo"]["missing_task_ids"], expected_ids)
            self.assertIsNone(models["grpo"]["reward"]["mean"])
            self.assertIsNone(models["grpo"]["reward"]["min"])
            self.assertEqual(sum(models["grpo"]["reward"]["histogram"]), 0)
            self.assertEqual(data["agreement"], [
                {"models": 0, "tasks": 199}, {"models": 1, "tasks": 1},
                {"models": 2, "tasks": 0}, {"models": 3, "tasks": 0},
            ])
            self.assertEqual(data["all_failed_task_ids"], expected_ids[1:])
            self.assertEqual(data["all_succeeded_task_ids"], [])
            self.assertEqual(data["best_model"], "baseline")

    def test_requires_identical_expected_task_sets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_run(root, "baseline", [1, 2], [])
            self._write_run(root, "sft", [1, 3], [])
            with self.assertRaisesRegex(ValueError, "same expected_task_ids"):
                build_comparison_data(root)

    def test_outcomes_cover_reward_v3_and_unknown_recordings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            categories = [*REWARD_V3_TYPES, "unknown"]
            rows = [
                {"task_id": task_id, "final_reward": 0.0,
                 "terminal_result": {"reward_detail": {"reward_type": reward_type}}}
                for task_id, reward_type in enumerate(categories, 1)
            ]
            self._write_run(root, "baseline", list(range(1, len(rows) + 2)), rows)

            data = build_comparison_data(root)

            self.assertEqual(data["outcome_order"], categories)
            self.assertEqual(data["models"][0]["outcomes"], dict.fromkeys(categories, 1))
            self.assertEqual(data["models"][0]["missing"], 1)

    def test_all_missing_and_empty_benchmarks_have_no_reward_distribution(self):
        for expected_ids in (list(range(1, 201)), []):
            with self.subTest(tasks=len(expected_ids)), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self._write_run(root, "baseline", expected_ids, [])

                data = build_comparison_data(root)

                model = data["models"][0]
                self.assertEqual(model["tasks"], len(expected_ids))
                self.assertEqual(model["recorded"], 0)
                self.assertEqual(model["missing_task_ids"], expected_ids)
                self.assertIsNone(model["reward"]["mean"])
                self.assertEqual(sum(model["reward"]["histogram"]), 0)
                self.assertEqual(data["all_failed_task_ids"], expected_ids)
                self.assertEqual(data["agreement"], [
                    {"models": 0, "tasks": len(expected_ids)}, {"models": 1, "tasks": 0},
                ])

    def test_requires_current_summary_fields(self):
        for field in ("expected_task_ids", "expected_tasks", "strict_success_task_ids"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run = self._write_run(root, "baseline", [1], [])
                summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
                del summary[field]
                (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
                with self.assertRaises(KeyError):
                    build_comparison_data(root)


if __name__ == "__main__":
    unittest.main()
