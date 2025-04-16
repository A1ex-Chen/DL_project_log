def text_proj_original_checkpoint_to_diffusers_checkpoint(checkpoint):
    diffusers_checkpoint = {'encoder_hidden_states_proj.weight': checkpoint
        [f'{DECODER_ORIGINAL_PREFIX}.text_seq_proj.0.weight'],
        'encoder_hidden_states_proj.bias': checkpoint[
        f'{DECODER_ORIGINAL_PREFIX}.text_seq_proj.0.bias'],
        'text_encoder_hidden_states_norm.weight': checkpoint[
        f'{DECODER_ORIGINAL_PREFIX}.text_seq_proj.1.weight'],
        'text_encoder_hidden_states_norm.bias': checkpoint[
        f'{DECODER_ORIGINAL_PREFIX}.text_seq_proj.1.bias'],
        'clip_extra_context_tokens_proj.weight': checkpoint[
        f'{DECODER_ORIGINAL_PREFIX}.clip_tok_proj.weight'],
        'clip_extra_context_tokens_proj.bias': checkpoint[
        f'{DECODER_ORIGINAL_PREFIX}.clip_tok_proj.bias'],
        'embedding_proj.weight': checkpoint[
        f'{DECODER_ORIGINAL_PREFIX}.text_feat_proj.weight'],
        'embedding_proj.bias': checkpoint[
        f'{DECODER_ORIGINAL_PREFIX}.text_feat_proj.bias'],
        'learned_classifier_free_guidance_embeddings': checkpoint[
        f'{DECODER_ORIGINAL_PREFIX}.cf_param'],
        'clip_image_embeddings_project_to_time_embeddings.weight':
        checkpoint[f'{DECODER_ORIGINAL_PREFIX}.clip_emb.weight'],
        'clip_image_embeddings_project_to_time_embeddings.bias': checkpoint
        [f'{DECODER_ORIGINAL_PREFIX}.clip_emb.bias']}
    return diffusers_checkpoint
