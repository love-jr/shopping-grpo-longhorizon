import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from scripts import prepare_opsd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "environments/ShopSimulator/shop_env"))

from web_agent_site.engine.reward_features import compile_reward_features  # noqa: E402


class PrepareOPSDTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "data/evaluation").mkdir(parents=True)
        (self.root / "data/evaluation/tasks.jsonl").write_text('{"task_id": 9999}\n')
        (self.root / "data/environment.json").write_text('{"product_data_sha256": "frozen"}')
        self.tasks = self.root / "tasks.jsonl"
        self.output = self.root / "contracts.json"
        self.product = {
            "Title": "镜子", "category": "镜子", "pricing": [20.0],
            "customization_options": {
                "颜色": [{"value": "红色", "price": 20.0}, {"value": "蓝色", "price": 10.0}],
                "尺寸": [{"value": "小号", "price": None}, {"value": "大号", "price": None}],
            },
            "options": {"颜色": ["红色", "蓝色"], "尺寸": ["小号", "大号"]},
        }

    def goal(self, options):
        instruction = {"instruction": "买一面镜子", "instruction_options": options, "attributes": []}
        return {
            "asin": "private", "instruction_text": instruction["instruction"],
            "attributes": [], "goal_options": options, "price_upper": None,
            **compile_reward_features(instruction, self.product),
        }

    def generate(self, goals):
        self.tasks.write_text("".join(json.dumps({"task_id": i}) + "\n" for i in range(len(goals))))
        engine = ModuleType("web_agent_site.engine.engine")
        engine.load_products = lambda path: ([self.product], {"private": self.product}, {}, None)
        goal_module = ModuleType("web_agent_site.engine.goal")
        goal_module.get_goals = lambda products, prices: goals
        # Catalog loading is isolated; option matching and price resolution are real.
        # Reward cannot be imported or used to construct the teacher reference.
        modules = {
            engine.__name__: engine,
            goal_module.__name__: goal_module,
            "web_agent_site.engine.reward": None,
        }
        with patch.object(prepare_opsd, "ROOT", self.root), patch.object(
            sys, "argv", ["prepare_opsd.py", "--tasks", str(self.tasks), "--output", str(self.output)]
        ), patch.object(sys, "path", list(sys.path)), patch.dict(sys.modules, modules):
            prepare_opsd.main()
        return json.loads(self.output.read_text())["contracts"]

    def test_explicit_selection_never_fills_unspecified_axes(self):
        contract = self.generate([self.goal(["红色"])])["0"]
        purchase = contract["reference_purchase"]
        self.assertEqual(purchase["selected_options"], {"颜色": "红色"})
        self.assertEqual(purchase["variant_price"], 20.0)
        self.assertEqual(contract["annotations"]["option_values"], ["红色"])
        self.assertNotIn("verification", purchase)
        self.assertNotIn("inferred_options", purchase)
        self.assertNotIn("reward", json.dumps(contract))

    def test_unmatched_annotation_is_preserved_without_cheapest_selection(self):
        contract = self.generate([self.goal(["银色八英寸?三色光"])])["0"]
        self.assertEqual(contract["reference_purchase"]["selected_options"], {})
        self.assertIsNone(contract["reference_purchase"]["variant_price"])
        self.assertEqual(contract["annotations"]["option_values"], ["银色八英寸?三色光"])
        self.assertEqual(contract["annotations"]["unresolved_options"][0]["reason"], "axis_not_found")

    def test_matching_failure_discards_partial_selection(self):
        goal = self.goal(["红色", "超大号"])
        goal["required_options_by_key"]["dimensions"] = {"value": "超大号"}
        contract = self.generate([goal])["0"]
        self.assertEqual(contract["reference_purchase"]["selected_options"], {})
        self.assertIsNone(contract["reference_purchase"]["variant_price"])

    def test_no_options_and_unknown_price_keep_all_thousand_tasks(self):
        contracts = self.generate([self.goal([]) for _ in range(1000)])
        self.assertEqual(set(contracts), {str(i) for i in range(1000)})
        for contract in contracts.values():
            self.assertEqual(contract["reference_purchase"]["selected_options"], {})
            self.assertIsNone(contract["reference_purchase"]["variant_price"])
            self.assertIsNone(contract["annotations"]["parsed_budget_upper"])

    def test_missing_query_or_title_fails_instead_of_filtering(self):
        goal = self.goal([])
        goal["instruction_text"] = ""
        with self.assertRaisesRegex(ValueError, "user_request"):
            self.generate([goal])
        self.product["Title"] = ""
        with self.assertRaisesRegex(ValueError, "title"):
            self.generate([self.goal([])])


if __name__ == "__main__":
    unittest.main()
