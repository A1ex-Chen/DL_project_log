def unet_original_checkpoint_to_diffusers_checkpoint(model, checkpoint):
    diffusers_checkpoint = {}
    num_head_channels = UNET_CONFIG['attention_head_dim']
    diffusers_checkpoint.update(unet_time_embeddings(checkpoint))
    diffusers_checkpoint.update(unet_conv_in(checkpoint))
    diffusers_checkpoint.update(unet_add_embedding(checkpoint))
    diffusers_checkpoint.update(unet_encoder_hid_proj(checkpoint))
    original_down_block_idx = 1
    for diffusers_down_block_idx in range(len(model.down_blocks)):
        checkpoint_update, num_original_down_blocks = (
            unet_downblock_to_diffusers_checkpoint(model, checkpoint,
            diffusers_down_block_idx=diffusers_down_block_idx,
            original_down_block_idx=original_down_block_idx,
            num_head_channels=num_head_channels))
        original_down_block_idx += num_original_down_blocks
        diffusers_checkpoint.update(checkpoint_update)
    diffusers_checkpoint.update(unet_midblock_to_diffusers_checkpoint(model,
        checkpoint, num_head_channels=num_head_channels))
    original_up_block_idx = 0
    for diffusers_up_block_idx in range(len(model.up_blocks)):
        checkpoint_update, num_original_up_blocks = (
            unet_upblock_to_diffusers_checkpoint(model, checkpoint,
            diffusers_up_block_idx=diffusers_up_block_idx,
            original_up_block_idx=original_up_block_idx, num_head_channels=
            num_head_channels))
        original_up_block_idx += num_original_up_blocks
        diffusers_checkpoint.update(checkpoint_update)
    diffusers_checkpoint.update(unet_conv_norm_out(checkpoint))
    diffusers_checkpoint.update(unet_conv_out(checkpoint))
    return diffusers_checkpoint
