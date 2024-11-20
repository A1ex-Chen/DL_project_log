def unet_encoder_hid_proj(checkpoint):
    diffusers_checkpoint = {}
    diffusers_checkpoint.update({'encoder_hid_proj.image_embeds.weight':
        checkpoint['clip_to_seq.weight'],
        'encoder_hid_proj.image_embeds.bias': checkpoint['clip_to_seq.bias'
        ], 'encoder_hid_proj.text_proj.weight': checkpoint[
        'to_model_dim_n.weight'], 'encoder_hid_proj.text_proj.bias':
        checkpoint['to_model_dim_n.bias']})
    return diffusers_checkpoint
