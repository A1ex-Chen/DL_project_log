def unet_time_embeddings(checkpoint):
    diffusers_checkpoint = {}
    diffusers_checkpoint.update({'time_embedding.linear_1.weight':
        checkpoint['time_embed.0.weight'], 'time_embedding.linear_1.bias':
        checkpoint['time_embed.0.bias'], 'time_embedding.linear_2.weight':
        checkpoint['time_embed.2.weight'], 'time_embedding.linear_2.bias':
        checkpoint['time_embed.2.bias']})
    return diffusers_checkpoint
