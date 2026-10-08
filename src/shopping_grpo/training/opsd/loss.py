"""Initialize current-batch normalization before veRL's native OPSD loss."""


def distillation_ppo_loss(
    config,
    distillation_config,
    model_output=None,
    data=None,
    dp_group=None,
    student_logits=None,
    data_format="thd",
):
    from verl.trainer.distillation.losses import distillation_ppo_loss as native_loss

    if student_logits is None:
        # Native distillation reads this before ppo_loss refreshes it.
        config.global_batch_info["dp_size"] = data["dp_size"]
        config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
        config.global_batch_info["global_batch_size"] = data["global_batch_size"]
        config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    return native_loss(
        config=config,
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
        dp_group=dp_group,
        student_logits=student_logits,
        data_format=data_format,
    )


def install_opsd_loss():
    """Ray worker setup: retain padding hooks, then replace only the worker alias."""
    from shopping_grpo.training.grpo.compat import install_torch_padding_fallback

    install_torch_padding_fallback()

    from verl.workers import engine_workers

    # ActorRolloutRefWorker.init_model binds this alias into its loss partial.
    engine_workers.distillation_ppo_loss = distillation_ppo_loss
