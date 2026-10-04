"""Shopping GRPO 终局奖励的纯函数测试。"""

import unittest

from shopping_grpo.training.grpo.adapter.runtime import (
    make_runtime_state,
    record_action_attempt,
    reward_breakdown,
)


def terminal_state(*, steps=8, native_reward=1.0, reward_type="gold_purchase"):
    state = make_runtime_state(task_id=1, max_steps=35)
    state["steps"] = [{"index": index} for index in range(steps)]
    state.update(
        {
            "done": True,
            "terminal_result": {"done": True, "over": True},
            "final_reward": native_reward,
            "reward_version": "shopsimulator-reward-v3",
            "reward_type": reward_type,
            "reward_detail": {
                "weighted_score": 1.0,
                "evidence_coverage": 1.0,
                "dimension_scores": {"key_options": 1.0},
                "hard_gates": {
                    "category": {"passed": True},
                    "budget": {"passed": True},
                },
            },
        }
    )
    return state


class ShoppingRewardTest(unittest.TestCase):
    def test_terminal_utility_is_not_reshaped_by_steps_or_repeated_actions(self):
        for steps in (8, 35):
            with self.subTest(steps=steps):
                state = terminal_state(steps=steps)
                state["action_attempt_count"] = 3
                state["repeat_action_count"] = 2

                result = reward_breakdown(state)

                self.assertEqual(result["total"], state["final_reward"])
                self.assertEqual(result["terminal_utility"], state["final_reward"])
                self.assertFalse(result["sampling_invalid"])
                self.assertAlmostEqual(result["repeat_action_rate"], 2 / 3)

    def test_terminal_utility_requires_a_complete_environment_terminal(self):
        for done, terminal_done, over in ((False, True, True), (True, False, True), (True, True, False)):
            with self.subTest(done=done, terminal_done=terminal_done, over=over):
                state = terminal_state()
                state["done"] = done
                state["terminal_result"] = {"done": terminal_done, "over": over}

                result = reward_breakdown(state)

                self.assertEqual(result["native"], 0.0)
                self.assertEqual(result["terminal_utility"], 0.0)
                self.assertEqual(result["total"], 0.0)

    def test_valid_negative_utility_remains_a_learning_signal(self):
        state = terminal_state(native_reward=-0.85, reward_type="wrong_purchase")

        result = reward_breakdown(state)

        self.assertEqual(result["total"], -0.85)
        self.assertEqual(result["purchase_success"], 0.0)
        self.assertFalse(result["sampling_invalid"])

    def test_infrastructure_invalid_or_nonfinite_utility_has_no_learning_signal(self):
        for infrastructure_invalid, native_reward in ((True, 1.0), (False, float("nan"))):
            with self.subTest(infrastructure_invalid=infrastructure_invalid):
                state = terminal_state(native_reward=native_reward)
                state["infrastructure_invalid"] = infrastructure_invalid

                result = reward_breakdown(state)

                self.assertTrue(result["infrastructure_invalid"])
                self.assertTrue(result["sampling_invalid"])
                self.assertEqual(result["total"], 0.0)

    def test_same_action_on_same_page_within_three_attempts_is_repeated(self):
        state = make_runtime_state(task_id=1, max_steps=35)

        record_action_attempt(state, "search_products", {"query": "mug"}, "search page")
        record_action_attempt(state, "open_product", {"asin": "123"}, "search page")
        record_action_attempt(state, "search_products", {"query": "mug"}, "search page")

        self.assertEqual(state["action_attempt_count"], 3)
        self.assertEqual(state["repeat_action_count"], 1)
        self.assertAlmostEqual(reward_breakdown(state)["repeat_action_rate"], 1 / 3)

    def test_different_parameters_or_page_are_not_repeated(self):
        state = make_runtime_state(task_id=1, max_steps=35)

        record_action_attempt(state, "search_products", {"query": "mug"}, "page 1")
        record_action_attempt(state, "search_products", {"query": "cup"}, "page 1")
        record_action_attempt(state, "search_products", {"query": "mug"}, "page 2")

        self.assertEqual(state["repeat_action_count"], 0)

    def test_think_is_not_an_environment_action_attempt(self):
        state = make_runtime_state(task_id=1, max_steps=35)

        record_action_attempt(state, "think", {"note": "plan"}, "page")

        self.assertEqual(state["action_attempt_count"], 0)
        self.assertEqual(state["recent_action_signatures"], [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
