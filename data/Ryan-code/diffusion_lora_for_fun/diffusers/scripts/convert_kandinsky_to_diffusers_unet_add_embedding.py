def unet_add_embedding(checkpoint):
    diffusers_checkpoint = {}
    diffusers_checkpoint.update({'add_embedding.text_norm.weight':
        checkpoint['ln_model_n.weight'], 'add_embedding.text_norm.bias':
        checkpoint['ln_model_n.bias'], 'add_embedding.text_proj.weight':
        checkpoint['proj_n.weight'], 'add_embedding.text_proj.bias':
        checkpoint['proj_n.bias'], 'add_embedding.image_proj.weight':
        checkpoint['img_layer.weight'], 'add_embedding.image_proj.bias':
        checkpoint['img_layer.bias']})
    return diffusers_checkpoint
