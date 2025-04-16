def unet_time_embeddings(checkpoint, original_unet_prefix):
    diffusers_checkpoint = {}
    diffusers_checkpoint.update({'time_embedding.linear_1.weight':
        checkpoint[f'{original_unet_prefix}.time_embed.0.weight'],
        'time_embedding.linear_1.bias': checkpoint[
        f'{original_unet_prefix}.time_embed.0.bias'],
        'time_embedding.linear_2.weight': checkpoint[
        f'{original_unet_prefix}.time_embed.2.weight'],
        'time_embedding.linear_2.bias': checkpoint[
        f'{original_unet_prefix}.time_embed.2.bias']})
    return diffusers_checkpoint
