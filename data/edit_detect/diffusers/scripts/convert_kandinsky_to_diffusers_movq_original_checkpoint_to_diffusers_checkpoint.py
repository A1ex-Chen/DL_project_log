def movq_original_checkpoint_to_diffusers_checkpoint(model, checkpoint):
    diffusers_checkpoint = {}
    diffusers_checkpoint.update(movq_encoder_to_diffusers_checkpoint(model,
        checkpoint))
    diffusers_checkpoint.update({'quant_conv.weight': checkpoint[
        'quant_conv.weight'], 'quant_conv.bias': checkpoint['quant_conv.bias']}
        )
    diffusers_checkpoint.update({'quantize.embedding.weight': checkpoint[
        'quantize.embedding.weight']})
    diffusers_checkpoint.update({'post_quant_conv.weight': checkpoint[
        'post_quant_conv.weight'], 'post_quant_conv.bias': checkpoint[
        'post_quant_conv.bias']})
    diffusers_checkpoint.update(movq_decoder_to_diffusers_checkpoint(model,
        checkpoint))
    return diffusers_checkpoint
