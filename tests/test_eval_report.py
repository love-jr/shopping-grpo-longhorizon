import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_glm_report import build_data
from shopping_grpo.evaluation.summary import REWARD_V3_TYPES, summarize_trajectories


class EvaluationReportTest(unittest.TestCase):
    def _write_run(self, run_dir, expected_ids, rows):
        summary = summarize_trajectories(expected_ids, rows)
        summary["protocol"] = {"model": "shopping-agent", "max_steps": 35}
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (run_dir / "trajectories.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def _trajectory(self, task_id, reward_type, reward=0.0):
        return {
            "task_id": task_id,
            "status": "done",
            "done": True,
            "final_reward": reward,
            "steps": [],
            "blocked_tool_calls": [],
            "initial_result": {"instruction": "test"},
            "terminal_result": {
                "done": True,
                "over": True,
                "reward_detail": {
                    "reward_version": "shopsimulator-reward-v3",
                    "reward_type": reward_type,
                    "reward_valid": True,
                    "purchase_success": reward_type == "gold_purchase",
                    "termination_reason": reward_type,
                },
            },
        }

    def test_report_reads_any_evaluation_directory_and_model_name(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self._write_run(run_dir, [42], [self._trajectory(42, "gold_purchase", 1.0)])

            data = build_data(run_dir)

            self.assertEqual(data["meta"]["model"], "shopping-agent")
            self.assertEqual(data["meta"]["label"], run_dir.name)
            self.assertEqual(data["summary"]["total"], 1)
            self.assertEqual(data["summary"]["strict"], 1)
            self.assertEqual(data["summary"]["mean_reward"], 1.0)

    def test_report_includes_early_abstain_and_guard_per_task(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            trajectory = self._trajectory(42, "early_abstain")
            trajectory["blocked_tool_calls"] = [{"reason": "test_guard"}]
            self._write_run(run_dir, [42], [trajectory])

            data = build_data(run_dir)

            early_abstain = next(
                item for item in data["charts"]["outcomes"]
                if item["key"] == "early_abstain"
            )
            self.assertEqual(early_abstain["value"], 1)
            self.assertEqual(data["summary"]["guard_per_task"], 1.0)

    def test_missing_tasks_keep_fixed_denominator_without_zero_rewards(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            trajectory = self._trajectory(42, "gold_purchase", 1.0)
            trajectory["blocked_tool_calls"] = [{"reason": "test_guard"}]
            self._write_run(run_dir, list(range(1, 201)), [trajectory])

            data = build_data(run_dir)

            summary = data["summary"]
            self.assertEqual(summary["total"], 200)
            self.assertEqual(summary["recorded"], 1)
            self.assertEqual(summary["missing"], 199)
            self.assertEqual(summary["strict_rate"], 1 / 200)
            self.assertEqual(summary["done_rate"], 1 / 200)
            self.assertEqual(summary["purchase_rate"], 1 / 200)
            self.assertEqual(summary["guard_per_task"], 1 / 200)
            self.assertEqual(summary["mean_reward"], 1.0)
            self.assertEqual(summary["reward_min"], 1.0)
            self.assertEqual(len(data["rows"]), 1)
            self.assertEqual(data["charts"]["missing"][0]["value"], 199)
            self.assertEqual(sum(item["value"] for item in data["charts"]["outcomes"]), 1)
            self.assertEqual(sum(item["tasks"] for item in data["charts"]["steps"]), 1)

    def test_empty_strict_success_ids_do_not_count_invalid_gold_purchase(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            trajectory = self._trajectory(42, "gold_purchase", 1.0)
            trajectory["terminal_result"]["reward_detail"]["reward_valid"] = False
            self._write_run(run_dir, [42], [trajectory])

            data = build_data(run_dir)

            self.assertEqual(data["summary"]["strict"], 0)
            self.assertEqual(data["summary"]["strict_rate"], 0.0)
            self.assertFalse(data["rows"][0]["strict"])
            self.assertEqual(data["summary"]["reward_counts"], {"gold_purchase": 1})

    def test_outcome_chart_counts_every_reward_v3_type_and_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            categories = [*REWARD_V3_TYPES, "unknown"]
            rows = [
                self._trajectory(task_id, reward_type)
                for task_id, reward_type in enumerate(categories, 1)
            ]
            self._write_run(run_dir, list(range(1, len(rows) + 2)), rows)

            data = build_data(run_dir)

            outcomes = {item["key"]: item["value"] for item in data["charts"]["outcomes"]}
            self.assertEqual(outcomes, dict.fromkeys(categories, 1))
            self.assertEqual(data["summary"]["reward_counts"], outcomes)
            self.assertEqual(sum(outcomes.values()), data["summary"]["recorded"])
            self.assertEqual(data["charts"]["missing"][0]["value"], 1)
            self.assertNotIn("missing", outcomes)

    def test_empty_trajectories_show_missing_without_reward_statistics(self):
        for expected_ids in (list(range(1, 201)), []):
            with self.subTest(tasks=len(expected_ids)), tempfile.TemporaryDirectory() as directory:
                run_dir = Path(directory)
                self._write_run(run_dir, expected_ids, [])

                data = build_data(run_dir)

                self.assertEqual(data["summary"]["total"], len(expected_ids))
                self.assertEqual(data["summary"]["recorded"], 0)
                self.assertEqual(data["summary"]["missing_task_ids"], expected_ids)
                self.assertEqual(data["rows"], [])
                for field in ("mean_reward", "reward_min", "reward_max", "reward_median", "median_steps"):
                    self.assertIsNone(data["summary"][field])
                self.assertTrue(all(item["value"] == 0 for item in data["charts"]["outcomes"]))
                self.assertEqual(data["charts"]["missing"][0]["value"], len(expected_ids))

    def test_uses_latest_expected_trajectory_like_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            rows = [self._trajectory(42, "gold_purchase", 1.0),
                    self._trajectory(42, "early_abstain", -0.2),
                    self._trajectory(999, "gold_purchase", 1.0)]
            self._write_run(run_dir, [42], rows)

            data = build_data(run_dir)

            self.assertEqual(data["summary"]["recorded"], 1)
            self.assertEqual(data["summary"]["strict"], 0)
            self.assertEqual(data["summary"]["reward_counts"], {"early_abstain": 1})
            self.assertEqual(data["rows"][0]["reward"], -0.2)

    def test_requires_current_summary_fields(self):
        for field in ("expected_task_ids", "expected_tasks", "strict_success_task_ids"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                run_dir = Path(directory)
                self._write_run(run_dir, [42], [])
                summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
                del summary[field]
                (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
                with self.assertRaises(KeyError):
                    build_data(run_dir)


if __name__ == "__main__":
    unittest.main()
