def map_txt_to_clip_feature(clip_model, tokenizer, prompt):
    text_inputs = tokenizer(prompt, padding='max_length', max_length=
        tokenizer.model_max_length, return_tensors='pt')
    text_input_ids = text_inputs.input_ids
    if text_input_ids.shape[-1] > tokenizer.model_max_length:
        removed_text = tokenizer.batch_decode(text_input_ids[:, tokenizer.
            model_max_length:])
        logger.warning(
            f'The following part of your input was truncated because CLIP can only handle sequences up to {tokenizer.model_max_length} tokens: {removed_text}'
            )
        text_input_ids = text_input_ids[:, :tokenizer.model_max_length]
    text_embeddings = clip_model.get_text_features(text_input_ids.to(
        clip_model.device))
    text_embeddings = text_embeddings / torch.linalg.norm(text_embeddings,
        dim=-1, keepdim=True)
    text_embeddings = text_embeddings[:, None, :]
    return text_embeddings[0][0].cpu().detach().numpy()
