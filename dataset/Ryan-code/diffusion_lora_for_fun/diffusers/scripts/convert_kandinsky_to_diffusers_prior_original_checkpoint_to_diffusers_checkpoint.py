def prior_original_checkpoint_to_diffusers_checkpoint(model, checkpoint,
    clip_stats_checkpoint):
    diffusers_checkpoint = {}
    diffusers_checkpoint.update({'time_embedding.linear_1.weight':
        checkpoint[f'{PRIOR_ORIGINAL_PREFIX}.time_embed.0.weight'],
        'time_embedding.linear_1.bias': checkpoint[
        f'{PRIOR_ORIGINAL_PREFIX}.time_embed.0.bias']})
    diffusers_checkpoint.update({'proj_in.weight': checkpoint[
        f'{PRIOR_ORIGINAL_PREFIX}.clip_img_proj.weight'], 'proj_in.bias':
        checkpoint[f'{PRIOR_ORIGINAL_PREFIX}.clip_img_proj.bias']})
    diffusers_checkpoint.update({'embedding_proj.weight': checkpoint[
        f'{PRIOR_ORIGINAL_PREFIX}.text_emb_proj.weight'],
        'embedding_proj.bias': checkpoint[
        f'{PRIOR_ORIGINAL_PREFIX}.text_emb_proj.bias']})
    diffusers_checkpoint.update({'encoder_hidden_states_proj.weight':
        checkpoint[f'{PRIOR_ORIGINAL_PREFIX}.text_enc_proj.weight'],
        'encoder_hidden_states_proj.bias': checkpoint[
        f'{PRIOR_ORIGINAL_PREFIX}.text_enc_proj.bias']})
    diffusers_checkpoint.update({'positional_embedding': checkpoint[
        f'{PRIOR_ORIGINAL_PREFIX}.positional_embedding']})
    diffusers_checkpoint.update({'prd_embedding': checkpoint[
        f'{PRIOR_ORIGINAL_PREFIX}.prd_emb']})
    diffusers_checkpoint.update({'time_embedding.linear_2.weight':
        checkpoint[f'{PRIOR_ORIGINAL_PREFIX}.time_embed.2.weight'],
        'time_embedding.linear_2.bias': checkpoint[
        f'{PRIOR_ORIGINAL_PREFIX}.time_embed.2.bias']})
    for idx in range(len(model.transformer_blocks)):
        diffusers_transformer_prefix = f'transformer_blocks.{idx}'
        original_transformer_prefix = (
            f'{PRIOR_ORIGINAL_PREFIX}.transformer.resblocks.{idx}')
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
        f'{PRIOR_ORIGINAL_PREFIX}.final_ln.weight'], 'norm_out.bias':
        checkpoint[f'{PRIOR_ORIGINAL_PREFIX}.final_ln.bias']})
    diffusers_checkpoint.update({'proj_to_clip_embeddings.weight':
        checkpoint[f'{PRIOR_ORIGINAL_PREFIX}.out_proj.weight'],
        'proj_to_clip_embeddings.bias': checkpoint[
        f'{PRIOR_ORIGINAL_PREFIX}.out_proj.bias']})
    clip_mean, clip_std = clip_stats_checkpoint
    clip_mean = clip_mean[None, :]
    clip_std = clip_std[None, :]
    diffusers_checkpoint.update({'clip_mean': clip_mean, 'clip_std': clip_std})
    return diffusers_checkpoint
