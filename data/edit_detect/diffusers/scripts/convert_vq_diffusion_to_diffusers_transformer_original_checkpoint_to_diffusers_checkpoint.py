def transformer_original_checkpoint_to_diffusers_checkpoint(model, checkpoint):
    diffusers_checkpoint = {}
    transformer_prefix = 'transformer.transformer'
    diffusers_latent_image_embedding_prefix = 'latent_image_embedding'
    latent_image_embedding_prefix = f'{transformer_prefix}.content_emb'
    diffusers_checkpoint.update({
        f'{diffusers_latent_image_embedding_prefix}.emb.weight': checkpoint
        [f'{latent_image_embedding_prefix}.emb.weight'],
        f'{diffusers_latent_image_embedding_prefix}.height_emb.weight':
        checkpoint[f'{latent_image_embedding_prefix}.height_emb.weight'],
        f'{diffusers_latent_image_embedding_prefix}.width_emb.weight':
        checkpoint[f'{latent_image_embedding_prefix}.width_emb.weight']})
    for transformer_block_idx, transformer_block in enumerate(model.
        transformer_blocks):
        diffusers_transformer_block_prefix = (
            f'transformer_blocks.{transformer_block_idx}')
        transformer_block_prefix = (
            f'{transformer_prefix}.blocks.{transformer_block_idx}')
        diffusers_ada_norm_prefix = (
            f'{diffusers_transformer_block_prefix}.norm1')
        ada_norm_prefix = f'{transformer_block_prefix}.ln1'
        diffusers_checkpoint.update(
            transformer_ada_norm_to_diffusers_checkpoint(checkpoint,
            diffusers_ada_norm_prefix=diffusers_ada_norm_prefix,
            ada_norm_prefix=ada_norm_prefix))
        diffusers_attention_prefix = (
            f'{diffusers_transformer_block_prefix}.attn1')
        attention_prefix = f'{transformer_block_prefix}.attn1'
        diffusers_checkpoint.update(
            transformer_attention_to_diffusers_checkpoint(checkpoint,
            diffusers_attention_prefix=diffusers_attention_prefix,
            attention_prefix=attention_prefix))
        diffusers_ada_norm_prefix = (
            f'{diffusers_transformer_block_prefix}.norm2')
        ada_norm_prefix = f'{transformer_block_prefix}.ln1_1'
        diffusers_checkpoint.update(
            transformer_ada_norm_to_diffusers_checkpoint(checkpoint,
            diffusers_ada_norm_prefix=diffusers_ada_norm_prefix,
            ada_norm_prefix=ada_norm_prefix))
        diffusers_attention_prefix = (
            f'{diffusers_transformer_block_prefix}.attn2')
        attention_prefix = f'{transformer_block_prefix}.attn2'
        diffusers_checkpoint.update(
            transformer_attention_to_diffusers_checkpoint(checkpoint,
            diffusers_attention_prefix=diffusers_attention_prefix,
            attention_prefix=attention_prefix))
        diffusers_norm_block_prefix = (
            f'{diffusers_transformer_block_prefix}.norm3')
        norm_block_prefix = f'{transformer_block_prefix}.ln2'
        diffusers_checkpoint.update({
            f'{diffusers_norm_block_prefix}.weight': checkpoint[
            f'{norm_block_prefix}.weight'],
            f'{diffusers_norm_block_prefix}.bias': checkpoint[
            f'{norm_block_prefix}.bias']})
        diffusers_feedforward_prefix = (
            f'{diffusers_transformer_block_prefix}.ff')
        feedforward_prefix = f'{transformer_block_prefix}.mlp'
        diffusers_checkpoint.update(
            transformer_feedforward_to_diffusers_checkpoint(checkpoint,
            diffusers_feedforward_prefix=diffusers_feedforward_prefix,
            feedforward_prefix=feedforward_prefix))
    diffusers_norm_out_prefix = 'norm_out'
    norm_out_prefix = f'{transformer_prefix}.to_logits.0'
    diffusers_checkpoint.update({f'{diffusers_norm_out_prefix}.weight':
        checkpoint[f'{norm_out_prefix}.weight'],
        f'{diffusers_norm_out_prefix}.bias': checkpoint[
        f'{norm_out_prefix}.bias']})
    diffusers_out_prefix = 'out'
    out_prefix = f'{transformer_prefix}.to_logits.1'
    diffusers_checkpoint.update({f'{diffusers_out_prefix}.weight':
        checkpoint[f'{out_prefix}.weight'], f'{diffusers_out_prefix}.bias':
        checkpoint[f'{out_prefix}.bias']})
    return diffusers_checkpoint
