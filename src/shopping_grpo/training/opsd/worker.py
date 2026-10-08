"""Shopping teacher contexts on veRL's native distillation worker seam."""

import asyncio
import json
from pathlib import Path

import ray
import torch
from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AgentLoopWorker
from verl.utils.chat_template import apply_chat_template
from verl.utils.tokenizer import normalize_token_ids

from shopping_grpo.training.grpo.adapter.runtime import task_id_from_kwargs
from shopping_grpo.training.opsd.privilege import reference_rejection_reasons, teacher_messages


def align_teacher_response(ids, logprobs, prompt_length, teacher_prompt_length):
    """Discard teacher-only prefix scores and retain the student's original positions."""
    # veRL/vLLM drops the first prompt score and appends one sentinel.  The
    # first student response token therefore lives at teacher index P_t - 1.
    ids = ids[teacher_prompt_length - 1 :]
    logprobs = logprobs[teacher_prompt_length - 1 :]
    prefix_length = prompt_length - 1
    return (
        torch.cat((ids.new_zeros((prefix_length, ids.shape[-1])), ids)),
        torch.cat((logprobs.new_zeros((prefix_length, logprobs.shape[-1])), logprobs)),
    )


def single_teacher(config):
    """Return the one configured teacher, or fail loudly.

    OPSD appends the same private contract to the teacher's system prompt and
    veRL's teacher_key/data_source routing cannot be honoured, so a second
    teacher would silently steal half the batch.
    """
    teacher_models = config.distillation.teacher_models
    if len(teacher_models) != 1:
        raise ValueError(
            "shopping OPSD supports exactly one teacher model, got "
            f"{sorted(teacher_models)}"
        )
    return next(iter(teacher_models.values()))


class ShoppingOPSDWorker(AgentLoopWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        single_teacher(self.config)
        contracts = json.loads(Path(self.config.shopping_opsd.contracts_path).read_text())
        manifest = json.loads(Path(self.config.shopping_opsd.manifest_path).read_text())
        if contracts["product_data_sha256"] != manifest["product_data_sha256"]:
            raise ValueError("OPSD contracts belong to a different product catalog")
        self.contracts = contracts["contracts"]
        if not self.contracts:
            raise ValueError("OPSD requires nonempty usable contracts")
        for task_id, contract in self.contracts.items():
            reasons = reference_rejection_reasons(contract)
            if reasons:
                raise ValueError(f"OPSD task {task_id} has an unusable reference: {'; '.join(reasons)}")

    async def _compute_teacher_logprobs(
        self, output, prompt_ids, response_ids, validate, sample_kwargs=None,
    ):
        if validate:
            return
        messages = teacher_messages(
            sample_kwargs["raw_prompt"],
            self.contracts[str(task_id_from_kwargs(sample_kwargs))],
        )
        processing_class = self.processor if self.processor is not None else self.tokenizer
        teacher_prompt = await asyncio.to_thread(
            apply_chat_template,
            processing_class,
            messages,
            tools=[tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True)
                   for tool in self.tools],
            tokenize=True,
            add_generation_prompt=True,
            **self.config.data.get("apply_chat_template_kwargs", {}),
        )
        teacher_prompt = normalize_token_ids(teacher_prompt)
        teacher_config = next(iter(self.config.distillation.teacher_models.values()))
        if len(teacher_prompt) + len(response_ids) + 1 > teacher_config.inference.max_model_len:
            raise ValueError("Teacher context exceeds max_model_len; refusing to truncate")
        ids, logprobs = await self.teacher_server_manager.compute_teacher_logprobs_single(
            sequence_ids=teacher_prompt + response_ids,
        )
        ids, logprobs = align_teacher_response(
            ids, logprobs, len(prompt_ids), len(teacher_prompt),
        )
        output.extra_fields["teacher_ids"] = ids
        output.extra_fields["teacher_logprobs"] = logprobs


class ShoppingOPSDManager(AgentLoopManager):
    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = ray.remote(ShoppingOPSDWorker)
        super().__init__(*args, **kwargs)
