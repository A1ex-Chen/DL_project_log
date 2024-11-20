def vqvae_original_checkpoint_to_diffusers_checkpoint(model, checkpoint):
    diffusers_checkpoint = {}
    diffusers_checkpoint.update(vqvae_encoder_to_diffusers_checkpoint(model,
        checkpoint))
    diffusers_checkpoint.update({'quant_conv.weight': checkpoint[
        'quant_conv.weight'], 'quant_conv.bias': checkpoint['quant_conv.bias']}
        )
    diffusers_checkpoint.update({'quantize.embedding.weight': checkpoint[
        'quantize.embedding']})
    diffusers_checkpoint.update({'post_quant_conv.weight': checkpoint[
        'post_quant_conv.weight'], 'post_quant_conv.bias': checkpoint[
        'post_quant_conv.bias']})
    diffusers_checkpoint.update(vqvae_decoder_to_diffusers_checkpoint(model,
        checkpoint))
    return diffusers_checkpoint
