"""把一条 veRL trajectory 绑定到一个 ShopSimulator 环境租约。

session 负责异步训练框架与同步 HTTP 客户端之间的边界：在线程中执行网络调用，
在 coroutine-local context 中暴露当前环境和运行状态，并在任何退出路径释放租约。
"""

from __future__ import annotations

import asyncio

from shopping_grpo.environment.client import ShopAgentEnv
from shopping_grpo.environment.observation import render_structured_observation
from shopping_grpo.training.grpo.adapter.runtime import current_environment, current_runtime_state, make_runtime_state
from shopping_grpo.training.grpo.trace import canonical_purchase_target


async def _wait_for_thread(function, *args):
    """线程调用结束后才传播取消，避免 reset/release 与租约清理竞态。"""
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    cancellation = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancellation = exc
    result = task.result()
    if cancellation is not None:
        raise cancellation
    return result


class ShopSimulatorSession:
    """负责 reset、绑定 coroutine-local 状态，并保证 release。"""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:5700",
        timeout: int = 60,
        max_steps: int = 35,
        required_environment_version: str | None = None,
        env_factory=None,
    ):
        self.base_url = base_url
        self.timeout = int(timeout)
        self.max_steps = int(max_steps)
        self.required_environment_version = required_environment_version
        self.env_factory = env_factory or ShopAgentEnv
        self.env = None
        self.state = None
        self.trace_target = None
        self._environment_token = None
        self._state_token = None

    async def start(self, task_id: int) -> dict:
        """启动一条 trajectory，并把首个 observation 放进运行状态。"""
        if self.env is not None:
            raise RuntimeError("ShopSimulator session has already started")
        try:
            self.env = self.env_factory(base_url=self.base_url, timeout=self.timeout)
            self.env.include_trace_target = True
            # reset 返回前客户端还不知道 env_idx，必须等线程结束才能释放。
            initial = await _wait_for_thread(self.env.reset, int(task_id))
            self.state = make_runtime_state(task_id=task_id, max_steps=self.max_steps)
            private_target = initial.get("_trace_target")
            if private_target is not None:
                if not isinstance(private_target, dict):
                    raise ValueError("ShopSimulator _trace_target must be an object")
                self.trace_target = canonical_purchase_target(
                    private_target.get("asin"), private_target.get("options")
                )
            actual_version = initial.get("environment_version")
            if (
                self.required_environment_version is not None
                and actual_version != self.required_environment_version
            ):
                raise RuntimeError(
                    "ShopSimulator environment version mismatch: "
                    f"expected {self.required_environment_version!r}, got {actual_version!r}"
                )
            self.state["latest_observation"] = render_structured_observation(
                initial.get("observation_state")
            )
            self.state["environment_version"] = actual_version
            # ContextVar 绑定和恢复必须留在同一 coroutine，不能把 close 创建成 task。
            self._environment_token = current_environment.set(self.env)
            self._state_token = current_runtime_state.set(self.state)
            return self.state
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        """释放环境并恢复 ContextVar；释放失败仍会清理本地绑定。"""
        if self.env is None:
            return
        try:
            await _wait_for_thread(self.env.release)
        except Exception as exc:
            if self.state is not None:
                self.state["error"] = f"release_error:{exc.__class__.__name__}:{exc}"
            raise
        finally:
            if self._state_token is not None:
                current_runtime_state.reset(self._state_token)
            if self._environment_token is not None:
                current_environment.reset(self._environment_token)
            self.env = None
            self._state_token = None
            self._environment_token = None
