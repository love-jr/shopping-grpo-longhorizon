"""veRL 购物适配层契约测试。"""

import asyncio
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

from shopping_grpo.training.grpo.adapter.agent_loop import ShoppingToolAgentLoop
from shopping_grpo.training.grpo.adapter.runtime import (
    current_environment,
    current_runtime_state,
    make_runtime_state,
    reward_breakdown,
    task_id_from_kwargs,
    terminal_reward,
)
from shopping_grpo.training.grpo.adapter.session import ShopSimulatorSession
from shopping_grpo.training.grpo.adapter.tools import ShopSimulatorTool


def make_reset_result():
    return {
        "environment_version": "shopsimulator-environment-v2.1",
        "observation_state": {
            "observation_version": "shopping-observation-v2",
            "page_type": "search_home",
            "search_available": True,
            "actions": [],
        },
    }


def make_tool(name):
    schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Test-only {name} tool.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
    try:
        from verl.tools.schemas import OpenAIFunctionToolSchema
    except ImportError:
        tool_schema = schema
    else:
        tool_schema = OpenAIFunctionToolSchema.model_validate(schema)
    return ShopSimulatorTool({}, tool_schema)


class VerlAdapterRuntimeTest(unittest.TestCase):
    def test_incremental_tool_tokens_are_not_left_truncated_to_initial_prompt_limit(self):
        class CharacterTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return [ord(c) for message in messages for c in message["content"]]

        async def run():
            loop = object.__new__(ShoppingToolAgentLoop)
            loop.tokenizer = CharacterTokenizer()
            loop.processor = None
            loop.loop = asyncio.get_running_loop()
            loop.apply_chat_template_kwargs = {}
            loop.system_prompt = []
            loop.rollout_config = SimpleNamespace(prompt_length=64)
            text = "price: 99.99\n" + "正文" * 100 + '\n可点击的按钮: ["buy now"]'
            tokens = await loop.apply_chat_template([{"role": "tool", "content": text}], remove_system_prompt=True)
            self.assertEqual("".join(map(chr, tokens)), text)

        asyncio.run(run())

    def test_agent_loop_preserves_real_verl_metrics_and_exports_shopping_diagnostics(self):
        created = []

        class FakeEnv:
            def __init__(self, **kwargs):
                self.released = False
                created.append(self)

            def reset(self, task_id):
                return make_reset_result()

            def release(self):
                self.released = True

        async def fake_parent_run(_loop, sampling_params, **kwargs):
            state = current_runtime_state.get()
            state.update(
                {
                    "done": True,
                    "terminal_result": {"done": True, "over": True},
                    "termination_reason": "gold_purchase",
                    "final_reward": 1.0,
                    "reward_version": "shopsimulator-reward-v3",
                    "reward_type": "gold_purchase",
                    "reward_valid": True,
                    "reward_detail": {
                        "reward_version": "shopsimulator-reward-v3",
                        "reward_type": "gold_purchase",
                        "termination_reason": "gold_purchase",
                        "reward_valid": True,
                        "terminal_utility": 1.0,
                        "purchase_success": True,
                        "sampling_invalid": False,
                        "weighted_score": 1.0,
                        "evidence_coverage": 1.0,
                        "dimension_scores": {"key_options": 1.0},
                        "hard_gates": {
                            "category": {
                                "status": "pass",
                                "passed": True,
                                "verifiable": True,
                            },
                            "budget": {
                                "status": "pass",
                                "passed": True,
                                "verifiable": True,
                            },
                        },
                    },
                    "steps": [
                        {
                            "index": 0,
                            "tool": "search",
                            "parameters": {"query": "shoe"},
                            "done": False,
                            "reward": 0.0,
                        }
                    ],
                    "guard_rejection_reason_counts": {"asin_not_visible": 2},
                }
            )
            return AgentLoopOutput(
                prompt_ids=[1],
                response_ids=[2],
                response_mask=[1],
                reward_score=None,
                metrics=AgentLoopMetrics(generate_sequences=0.25),
                extra_fields={},
            )

        async def run():
            loop = object.__new__(ShoppingToolAgentLoop)
            loop.base_url = "http://shop.test"
            loop.timeout = 60
            loop.max_steps = 35
            loop.required_environment_version = "shopsimulator-environment-v2.1"
            loop.reward_mode = "constraint_aware"
            loop.env_factory = FakeEnv
            with patch.object(ToolAgentLoop, "run", fake_parent_run):
                return await ShoppingToolAgentLoop.run(
                    loop,
                    {},
                    extra_info={"task_id": 42},
                )

        output = asyncio.run(run())
        self.assertIsInstance(output.metrics, AgentLoopMetrics)
        self.assertEqual(
            output.metrics.model_dump(),
            {
                "generate_sequences": 0.25,
                "tool_calls": 0.0,
                "compute_score": 0.0,
                "num_preempted": -1,
            },
        )
        self.assertEqual(output.reward_score, 1.0)
        self.assertEqual(output.extra_fields["shopping"]["task_id"], 42)
        self.assertEqual(
            output.extra_fields["shopping"]["reward"]["terminal_utility"],
            1.0,
        )
        self.assertEqual(
            output.extra_fields["shopping"]["actions"],
            [{"tool": "search", "parameters": {"query": "shoe"}}],
        )
        self.assertEqual(
            output.extra_fields["shopping"]["guard_rejection_reasons"],
            {"asin_not_visible": 2},
        )
        self.assertTrue(created[0].released)

    def test_terminal_reward_only_uses_a_normal_environment_completion(self):
        done = make_runtime_state(task_id=1, max_steps=35)
        done.update({"done": True, "terminal_result": {"done": True, "over": True}, "final_reward": 0.75})
        self.assertEqual(terminal_reward(done), 0.75)

        unfinished = make_runtime_state(task_id=1, max_steps=35)
        unfinished.update({"final_reward": 1.0, "terminal_result": {"done": False}})
        self.assertEqual(terminal_reward(unfinished), 0.0)

        errored = make_runtime_state(task_id=1, max_steps=35)
        errored.update(
            {
                "done": True,
                "terminal_result": {"done": True, "over": True},
                "final_reward": 1.0,
                "error": "tool_error:timeout",
            }
        )
        self.assertEqual(terminal_reward(errored), 0.0)


    def test_task_id_is_read_from_verl_extra_info(self):
        self.assertEqual(task_id_from_kwargs({"extra_info": {"task_id": 42}}), 42)

    def test_task_id_accepts_numpy_style_scalar_container(self):
        class Scalar:
            def item(self):
                return {"task_id": 43}

        self.assertEqual(task_id_from_kwargs({"extra_info": Scalar()}), 43)

    def test_missing_task_id_fails_before_acquiring_an_environment(self):
        with self.assertRaisesRegex(ValueError, "task_id"):
            task_id_from_kwargs({"extra_info": {"split": "train"}})

    def test_invalid_terminal_reward_is_infrastructure_invalid_without_private_state(self):
        class FakeEnv:
            def __init__(self, overrides):
                self.overrides = overrides

            def step(self, action):
                return {
                    "instruction": "Goal: hidden answer\nReward: hidden breakdown",
                    "done": True,
                    "over": True,
                    "reward": 1.0,
                    "goal": {"hidden_answer": "private goal"},
                    "reward_detail": {
                        "reward_version": "shopsimulator-reward-v3",
                        "reward_type": "gold_purchase",
                        "termination_reason": "gold_purchase",
                        "reward_valid": True,
                        "terminal_utility": 1.0,
                        "purchase_success": True,
                        "sampling_invalid": False,
                        "hard_gates": {},
                        "hidden_answer": "private reward evidence",
                        **self.overrides,
                    },
                }

        async def run(overrides):
            state = make_runtime_state(task_id=2, max_steps=35)
            state["latest_observation"] = "搜索功能是否可用: True"
            env_token = current_environment.set(FakeEnv(overrides))
            state_token = current_runtime_state.set(state)
            try:
                response, _, diagnostics = await make_tool("search_products").execute(
                    "tool-1", {"query": "mug"}
                )
            finally:
                current_runtime_state.reset(state_token)
                current_environment.reset(env_token)
            self.assertTrue(state["terminate"])
            self.assertEqual(state["terminal_result"], {"done": True, "over": True})
            self.assertTrue(state["infrastructure_invalid"])
            self.assertIsNone(state["reward_detail"])
            self.assertNotIn("hidden", str((state, response.text, diagnostics)))
            self.assertNotIn("private", str((state, response.text, diagnostics)))
            breakdown = reward_breakdown(state)
            self.assertTrue(breakdown["sampling_invalid"])
            self.assertEqual(breakdown["terminal_utility"], 0.0)

        for overrides in (
            {"weighted_score": float("nan")},
            {"reward_valid": "true"},
            {"reward_valid": False, "sampling_invalid": True},
            {"terminal_utility": 0.55},
        ):
            with self.subTest(overrides=overrides):
                asyncio.run(run(overrides))

    def test_terminal_reward_keeps_unverifiable_separate_from_infrastructure(self):
        class FakeEnv:
            def step(self, action):
                return {
                    "instruction": "terminal",
                    "done": True,
                    "over": True,
                    "reward": 0.0,
                    "termination_reason": "reward_unverifiable",
                    "reward_valid": False,
                    "reward_detail": {
                        "reward_version": "shopsimulator-reward-v3",
                        "reward_type": "reward_unverifiable",
                        "reward_valid": False,
                        "termination_reason": "reward_unverifiable",
                        "target_asin_match": False,
                        "terminal_utility": 0.0,
                        "purchase_success": False,
                        "sampling_invalid": True,
                        "hard_gates": {
                            "category": {
                                "status": "unverifiable",
                                "passed": False,
                                "verifiable": False,
                            }
                        },
                        "weighted_score": 0.0,
                    },
                }

        async def run():
            state = make_runtime_state(task_id=2, max_steps=35)
            state["latest_observation"] = "搜索功能是否可用: True"
            env_token = current_environment.set(FakeEnv())
            state_token = current_runtime_state.set(state)
            try:
                await make_tool("search_products").execute(
                    "tool-v2", {"query": "mug"}
                )
            finally:
                current_runtime_state.reset(state_token)
                current_environment.reset(env_token)
            self.assertFalse(state["infrastructure_invalid"])
            self.assertTrue(state["reward_unverifiable"])
            self.assertEqual(state["reward_type"], "reward_unverifiable")
            self.assertEqual(state["termination_reason"], "reward_unverifiable")
            self.assertFalse(state["reward_valid"])
            breakdown = reward_breakdown(state)
            self.assertTrue(breakdown["sampling_invalid"])
            self.assertTrue(breakdown["reward_unverifiable"])
            self.assertFalse(breakdown["infrastructure_invalid"])
            self.assertEqual(breakdown["terminal_utility"], 0.0)

        asyncio.run(run())

    def test_reward_exposes_utility_success_and_sampling_validity_separately(self):
        class FakeEnv:
            def step(self, action):
                return {
                    "instruction": "Goal: hidden terminal answer",
                    "done": True,
                    "over": True,
                    "reward": 0.55,
                    "termination_reason": "valid_alternative_purchase",
                    "reward_valid": True,
                    "goal": {"hidden_answer": "private goal"},
                    "reward_detail": {
                        "reward_version": "shopsimulator-reward-v3",
                        "reward_type": "valid_alternative_purchase",
                        "reward_valid": True,
                        "termination_reason": "valid_alternative_purchase",
                        "target_asin_match": False,
                        "terminal_utility": 0.55,
                        "purchase_success": True,
                        "sampling_invalid": False,
                        "weighted_score": 1.0,
                        "evidence_coverage": 1.0,
                        "dimension_scores": {
                            "brand": 0.0,
                            "model": 0.0,
                            "core_functions": 1.0,
                            "key_options": 1.0,
                            "hidden_answer": "private dimension evidence",
                        },
                        "hard_gates": {
                            "category": {
                                "status": "pass",
                                "passed": True,
                                "verifiable": True,
                                "comparator": "category_leaf_ancestor_chain",
                                "source_field": "category",
                                "hidden_answer": "private gate evidence",
                            }
                        },
                        "hidden_answer": "private reward evidence",
                    },
                }

        async def run():
            state = make_runtime_state(task_id=2, max_steps=35)
            state["latest_observation"] = "搜索功能是否可用: True"
            env_token = current_environment.set(FakeEnv())
            state_token = current_runtime_state.set(state)
            try:
                response, _, diagnostics = await make_tool("search_products").execute(
                    "tool-v3",
                    {"query": "mug"},
                )
            finally:
                current_runtime_state.reset(state_token)
                current_environment.reset(env_token)
            self.assertFalse(state["infrastructure_invalid"])
            self.assertFalse(state["reward_unverifiable"])
            self.assertTrue(state["reward_valid"])
            self.assertEqual(state["latest_observation"], "搜索功能是否可用: True")
            self.assertEqual(
                state["reward_detail"]["dimension_scores"]["core_functions"],
                1.0,
            )
            self.assertNotIn("goal", state)
            self.assertNotIn("hidden", str((state, response.text, diagnostics)))
            self.assertNotIn("private", str((state, response.text, diagnostics)))
            self.assertEqual(
                state["reward_type"],
                "valid_alternative_purchase",
            )
            breakdown = reward_breakdown(state)
            self.assertEqual(breakdown["terminal_utility"], 0.55)
            self.assertEqual(breakdown["purchase_success"], 1.0)
            self.assertEqual(breakdown["r_att"], 1.0)
            self.assertEqual(breakdown["r_option"], 1.0)
            self.assertFalse(breakdown["sampling_invalid"])

        asyncio.run(run())


    def test_think_consumes_the_step_budget_and_terminates_at_the_exact_limit(self):
        async def run():
            state = make_runtime_state(task_id=2, max_steps=1)
            env_token = current_environment.set(object())
            state_token = current_runtime_state.set(state)
            try:
                response, _, _ = await make_tool("think").execute("tool-1", {"note": "plan"})
            finally:
                current_runtime_state.reset(state_token)
                current_environment.reset(env_token)
            self.assertEqual(len(state["steps"]), 1)
            self.assertTrue(state["terminate"])
            self.assertEqual(state["error"], "max_steps")
            self.assertIn("maximum", response.text)

        asyncio.run(run())

    def test_repeated_guard_rejections_terminate_instead_of_looping_forever(self):
        async def run():
            state = make_runtime_state(task_id=2, max_steps=35)
            state["latest_observation"] = "可点击的按钮: []"
            state["latest_observation_truncated"] = True
            env_token = current_environment.set(object())
            state_token = current_runtime_state.set(state)
            try:
                tool = make_tool("open_product")
                for index in range(3):
                    response, _, _ = await tool.execute(f"tool-{index}", {"asin": "123456789012"})
            finally:
                current_runtime_state.reset(state_token)
                current_environment.reset(env_token)
            self.assertTrue(state["terminate"])
            self.assertEqual(state["error"], "too_many_guard_rejections")
            self.assertEqual(state["steps"], [])
            self.assertEqual(state["action_attempt_count"], 3)
            self.assertEqual(state["repeat_action_count"], 2)
            self.assertEqual(state["guard_rejection_count"], 3)
            self.assertEqual(state["guard_rejection_after_truncation_count"], 3)
            self.assertEqual(state["action_attempt_after_truncation_count"], 3)
            self.assertIn("maximum", response.text)

        asyncio.run(run())

    def test_session_releases_its_environment_on_close(self):
        """无论正常终局还是异常路径，veRL lifecycle 都必须归还 ShopSimulator 租约。"""
        created = []

        class FakeEnv:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.released = False
                created.append(self)

            def reset(self, task_id):
                return make_reset_result()

            def release(self):
                self.released = True

        async def run():
            outer_environment = object()
            outer_state = {"outer": True}
            environment_token = current_environment.set(outer_environment)
            state_token = current_runtime_state.set(outer_state)
            try:
                session = ShopSimulatorSession(max_steps=35, env_factory=FakeEnv)
                state = await session.start(task_id=8)
                self.assertIs(current_environment.get(), created[0])
                self.assertIs(current_runtime_state.get(), state)
                self.assertIn("[SHOPPING_OBSERVATION_V2]", state["latest_observation"])
                state.update({"done": True, "terminal_result": {"done": True, "over": True}, "final_reward": 1.0})
                self.assertEqual(terminal_reward(state), 1.0)
                await session.close()
                self.assertIs(current_environment.get(), outer_environment)
                self.assertIs(current_runtime_state.get(), outer_state)
                self.assertIsNone(session.env)
            finally:
                current_runtime_state.reset(state_token)
                current_environment.reset(environment_token)

        asyncio.run(run())
        self.assertTrue(created[0].released)


    def test_session_render_failure_releases_and_preserves_outer_context(self):
        created = []

        class FakeEnv:
            def __init__(self, **kwargs):
                self.released = False
                created.append(self)

            def reset(self, task_id):
                result = make_reset_result()
                result["observation_state"]["page_type"] = "invalid_page"
                return result

            def release(self):
                self.released = True

        async def run():
            outer_environment = object()
            outer_state = {"outer": True}
            environment_token = current_environment.set(outer_environment)
            state_token = current_runtime_state.set(outer_state)
            try:
                session = ShopSimulatorSession(env_factory=FakeEnv)
                with self.assertRaisesRegex(ValueError, "unsupported page_type"):
                    await session.start(8)
                self.assertIsNone(session.env)
                self.assertIs(current_environment.get(), outer_environment)
                self.assertIs(current_runtime_state.get(), outer_state)
            finally:
                current_runtime_state.reset(state_token)
                current_environment.reset(environment_token)

        asyncio.run(run())
        self.assertTrue(created[0].released)

    def test_session_rejects_legacy_initial_observation_and_releases(self):
        async def run(initial):
            class FakeEnv:
                def __init__(self, **kwargs):
                    self.released = False

                def reset(self, task_id):
                    return initial

                def release(self):
                    self.released = True

            env = FakeEnv()
            session = ShopSimulatorSession(env_factory=lambda **kwargs: env)
            with self.assertRaises((AttributeError, ValueError)):
                await session.start(8)
            self.assertTrue(env.released)
            self.assertIsNone(session.env)

        for initial in ("legacy observation", {"instruction": "legacy observation"}):
            with self.subTest(initial=initial):
                asyncio.run(run(initial))

    def test_session_invalid_private_target_releases(self):
        async def run(private_target):
            class FakeEnv:
                def __init__(self, **kwargs):
                    self.released = False

                def reset(self, task_id):
                    result = make_reset_result()
                    result["_trace_target"] = private_target
                    return result

                def release(self):
                    self.released = True

            env = FakeEnv()
            session = ShopSimulatorSession(env_factory=lambda **kwargs: env)
            with self.assertRaises(ValueError):
                await session.start(8)
            self.assertTrue(env.released)
            self.assertIsNone(session.env)

        for private_target in ("not an object", {"asin": "", "options": {}}):
            with self.subTest(private_target=private_target):
                asyncio.run(run(private_target))

    def test_cancelled_reset_waits_for_lease_response_before_release(self):
        async def run():
            loop = asyncio.get_running_loop()
            reset_started = asyncio.Event()
            finish_reset = threading.Event()
            events = []

            class FakeEnv:
                def __init__(self, **kwargs):
                    self.env_idx = None

                def reset(self, task_id):
                    events.append("reset_started")
                    loop.call_soon_threadsafe(reset_started.set)
                    if not finish_reset.wait(5):
                        raise RuntimeError("test did not unblock reset")
                    self.env_idx = 42
                    events.append("reset_finished")
                    return make_reset_result()

                def release(self):
                    events.append(("release", self.env_idx))
                    self.env_idx = None

            session = ShopSimulatorSession(env_factory=FakeEnv)
            outer_environment = current_environment.get()
            outer_state = current_runtime_state.get()

            async def start():
                try:
                    await session.start(8)
                finally:
                    self.assertIs(current_environment.get(), outer_environment)
                    self.assertIs(current_runtime_state.get(), outer_state)

            worker = asyncio.create_task(start())
            try:
                await asyncio.wait_for(reset_started.wait(), 5)
                worker.cancel()
                await asyncio.sleep(0)
                worker.cancel()
                await asyncio.sleep(0)
                self.assertFalse(worker.done())
                self.assertEqual(events, ["reset_started"])
            finally:
                finish_reset.set()
            with self.assertRaises(asyncio.CancelledError):
                await worker
            self.assertEqual(events, ["reset_started", "reset_finished", ("release", 42)])
            self.assertIsNone(session.env)

        asyncio.run(run())

    def test_cancelled_close_waits_for_release_and_restores_original_context(self):
        async def run():
            loop = asyncio.get_running_loop()
            release_started = asyncio.Event()
            finish_release = threading.Event()
            events = []

            class FakeEnv:
                def __init__(self, **kwargs):
                    pass

                def reset(self, task_id):
                    return make_reset_result()

                def release(self):
                    events.append("release_started")
                    loop.call_soon_threadsafe(release_started.set)
                    if not finish_release.wait(5):
                        raise RuntimeError("test did not unblock release")
                    events.append("release_finished")

            session = ShopSimulatorSession(env_factory=FakeEnv)
            outer_environment = object()
            outer_state = {"outer": True}

            async def start_and_close():
                environment_token = current_environment.set(outer_environment)
                state_token = current_runtime_state.set(outer_state)
                try:
                    await session.start(8)
                    self.assertIs(current_environment.get(), session.env)
                    self.assertIs(current_runtime_state.get(), session.state)
                    try:
                        await session.close()
                    finally:
                        self.assertIs(current_environment.get(), outer_environment)
                        self.assertIs(current_runtime_state.get(), outer_state)
                finally:
                    current_runtime_state.reset(state_token)
                    current_environment.reset(environment_token)

            worker = asyncio.create_task(start_and_close())
            try:
                await asyncio.wait_for(release_started.wait(), 5)
                worker.cancel()
                await asyncio.sleep(0)
                worker.cancel()
                await asyncio.sleep(0)
                self.assertFalse(worker.done())
                self.assertEqual(events, ["release_started"])
                self.assertIsNotNone(session.env)
            finally:
                finish_release.set()
            with self.assertRaises(asyncio.CancelledError):
                await worker
            self.assertEqual(events, ["release_started", "release_finished"])
            self.assertIsNone(session.env)
            self.assertIsNone(session.state["error"])

        asyncio.run(run())

    def test_session_rejects_wrong_environment_version_and_releases(self):
        created = []

        class FakeEnv:
            def __init__(self, **kwargs):
                self.released = False
                created.append(self)

            def reset(self, task_id):
                result = make_reset_result()
                result["environment_version"] = "unsupported-environment"
                return result

            def release(self):
                self.released = True

        async def run():
            session = ShopSimulatorSession(
                required_environment_version="shopsimulator-environment-v2.1",
                env_factory=FakeEnv,
            )
            with self.assertRaisesRegex(RuntimeError, "version mismatch"):
                await session.start(1)

        asyncio.run(run())
        self.assertTrue(created[0].released)

    def test_reset_failure_still_releases_the_environment(self):
        created = []

        class FakeEnv:
            def __init__(self, **kwargs):
                self.released = False
                created.append(self)

            def reset(self, task_id):
                raise RuntimeError("reset failed")

            def release(self):
                self.released = True

        async def run():
            session = ShopSimulatorSession(env_factory=FakeEnv)
            with self.assertRaisesRegex(RuntimeError, "reset failed"):
                await session.start(task_id=8)

        asyncio.run(run())
        self.assertTrue(created[0].released)

    def test_release_failure_is_not_silently_hidden_or_forgotten(self):
        class FakeEnv:
            def __init__(self, **kwargs):
                pass

            def reset(self, task_id):
                return make_reset_result()

            def release(self):
                raise RuntimeError("release failed")

        async def run():
            session = ShopSimulatorSession(env_factory=FakeEnv)
            await session.start(task_id=8)
            with self.assertRaisesRegex(RuntimeError, "release failed"):
                await session.close()
            self.assertEqual(session.state["error"], "release_error:RuntimeError:release failed")
            self.assertIsNone(session.env)
            self.assertIsNone(current_environment.get())
            self.assertIsNone(current_runtime_state.get())

        asyncio.run(run())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
