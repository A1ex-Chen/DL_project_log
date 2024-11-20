def prior_image_original_checkpoint_to_diffusers_checkpoint(model, checkpoint):
    diffusers_checkpoint = {}
    diffusers_checkpoint.update({'time_embedding.linear_1.weight':
        checkpoint[f'{PRIOR_IMAGE_ORIGINAL_PREFIX}.time_embed.c_fc.weight'],
        'time_embedding.linear_1.bias': checkpoint[
        f'{PRIOR_IMAGE_ORIGINAL_PREFIX}.time_embed.c_fc.bias']})
    diffusers_checkpoint.update({'time_embedding.linear_2.weight':
        checkpoint[
        f'{PRIOR_IMAGE_ORIGINAL_PREFIX}.time_embed.c_proj.weight'],
        'time_embedding.linear_2.bias': checkpoint[
        f'{PRIOR_IMAGE_ORIGINAL_PREFIX}.time_embed.c_proj.bias']})
    diffusers_checkpoint.update({'proj_in.weight': checkpoint[
        f'{PRIOR_IMAGE_ORIGINAL_PREFIX}.input_proj.weight'], 'proj_in.bias':
        checkpoint[f'{PRIOR_IMAGE_ORIGINAL_PREFIX}.input_proj.bias']})
    diffusers_checkpoint.update({'embedding_proj_norm.weight': checkpoint[
        f'{PRIOR_IMAGE_ORIGINAL_PREFIX}.clip_embed.0.weight'],
        'embedding_proj_norm.bias': checkpoint[
        f'{PRIOR_IMAGE_ORIGINAL_PREFIX}.clip_embed.0.bias']})
    diffusers_checkpoint.update({'embedding_proj.weight': checkpoint[
        f'{PRIOR_IMAGE_ORIGINAL_PREFIX}.clip_embed.1.weight'],
        'embedding_proj.bias': checkpoint[
        f'{PRIOR_IMAGE_ORIGINAL_PREFIX}.clip_embed.1.bias']})
    diffusers_checkpoint.update({'positional_embedding': checkpoint[
        f'{PRIOR_IMAGE_ORIGINAL_PREFIX}.pos_emb'][None, :]})
    diffusers_checkpoint.update({'norm_in.weight': checkpoint[
        f'{PRIOR_IMAGE_ORIGINAL_PREFIX}.ln_pre.weight'], 'norm_in.bias':
        checkpoint[f'{PRIOR_IMAGE_ORIGINAL_PREFIX}.ln_pre.bias']})
    for idx in range(len(model.transformer_blocks)):
        diffusers_transformer_prefix = f'transformer_blocks.{idx}'
        original_transformer_prefix = (
            f'{PRIOR_IMAGE_ORIGINAL_PREFIX}.backbone.resblocks.{idx}')
        diffusers_attention_prefix = f'{diffusers_transformer_prefix}.attn1'
        original_attention_prefix = f'{original_transformer_prefix}.attn'
        diffusers_checkpoint.update(prior_attention_to_diffusers(checkpoint,
            diffusers_attention_prefix=diffusers_attention_prefix,
            original_attention_prefix=original_attention_prefix,
            attention_head_dim=model.attention_head_dim))
        diffusers_ff_prefix = f'{diffusers_transformer_prefix}.ff'
        original_ff_prefix = f'{original_transformer_prefix}.mlp'
        diffusers_checkpoint.update(prior_ff_to_diffusers(checkpoint,
            diffusers_ff_prefix=diffusers_ff_prefix, original_ff_prefix=
            original_ff_prefix))
        diffusers_checkpoint.update({
            f'{diffusers_transformer_prefix}.norm1.weight': checkpoint[
            f'{original_transformer_prefix}.ln_1.weight'],
            f'{diffusers_transformer_prefix}.norm1.bias': checkpoint[
            f'{original_transformer_prefix}.ln_1.bias']})
        diffusers_checkpoint.update({
            f'{diffusers_transformer_prefix}.norm3.weight': checkpoint[
            f'{original_transformer_prefix}.ln_2.weight'],
            f'{diffusers_transformer_prefix}.norm3.bias': checkpoint[
            f'{original_transformer_prefix}.ln_2.bias']})
    diffusers_checkpoint.update({'norm_out.weight': checkpoint[
        f'{PRIOR_IMAGE_ORIGINAL_PREFIX}.ln_post.weight'], 'norm_out.bias':
        checkpoint[f'{PRIOR_IMAGE_ORIGINAL_PREFIX}.ln_post.bias']})
    diffusers_checkpoint.update({'proj_to_clip_embeddings.weight':
        checkpoint[f'{PRIOR_IMAGE_ORIGINAL_PREFIX}.output_proj.weight'],
        'proj_to_clip_embeddings.bias': checkpoint[
        f'{PRIOR_IMAGE_ORIGINAL_PREFIX}.output_proj.bias']})
    return diffusers_checkpoint
