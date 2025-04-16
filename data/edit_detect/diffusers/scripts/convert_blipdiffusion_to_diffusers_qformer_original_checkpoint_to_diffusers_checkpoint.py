def qformer_original_checkpoint_to_diffusers_checkpoint(model):
    qformer_checkpoint = {}
    qformer_checkpoint.update(embeddings_from_original_checkpoint(model,
        'embeddings', 'blip.Qformer.bert.embeddings'))
    qformer_checkpoint.update({'query_tokens': model['blip.query_tokens']})
    qformer_checkpoint.update(proj_layer_from_original_checkpoint(model,
        'proj_layer', 'proj_layer'))
    qformer_checkpoint.update(encoder_from_original_checkpoint(model,
        'encoder.layer', 'blip.Qformer.bert.encoder.layer'))
    qformer_checkpoint.update(visual_encoder_from_original_checkpoint(model,
        'visual_encoder', 'blip.visual_encoder'))
    return qformer_checkpoint
