def super_res_unet_last_step_original_checkpoint_to_diffusers_checkpoint(model,
    checkpoint):
    diffusers_checkpoint = {}
    original_unet_prefix = SUPER_RES_UNET_LAST_STEP_PREFIX
    diffusers_checkpoint.update(unet_time_embeddings(checkpoint,
        original_unet_prefix))
    diffusers_checkpoint.update(unet_conv_in(checkpoint, original_unet_prefix))
    original_down_block_idx = 1
    for diffusers_down_block_idx in range(len(model.down_blocks)):
        checkpoint_update, num_original_down_blocks = (
            unet_downblock_to_diffusers_checkpoint(model, checkpoint,
            diffusers_down_block_idx=diffusers_down_block_idx,
            original_down_block_idx=original_down_block_idx,
            original_unet_prefix=original_unet_prefix, num_head_channels=None))
        original_down_block_idx += num_original_down_blocks
        diffusers_checkpoint.update(checkpoint_update)
    diffusers_checkpoint.update(unet_midblock_to_diffusers_checkpoint(model,
        checkpoint, original_unet_prefix=original_unet_prefix,
        num_head_channels=None))
    original_up_block_idx = 0
    for diffusers_up_block_idx in range(len(model.up_blocks)):
        checkpoint_update, num_original_up_blocks = (
            unet_upblock_to_diffusers_checkpoint(model, checkpoint,
            diffusers_up_block_idx=diffusers_up_block_idx,
            original_up_block_idx=original_up_block_idx,
            original_unet_prefix=original_unet_prefix, num_head_channels=None))
        original_up_block_idx += num_original_up_blocks
        diffusers_checkpoint.update(checkpoint_update)
    diffusers_checkpoint.update(unet_conv_norm_out(checkpoint,
        original_unet_prefix))
    diffusers_checkpoint.update(unet_conv_out(checkpoint, original_unet_prefix)
        )
    return diffusers_checkpoint
