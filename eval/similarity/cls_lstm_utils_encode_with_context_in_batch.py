def encode_with_context_in_batch(model, batch_size, text_list, context_list,
    device):
    all_contextual_emb_tensor_list = []
    if isinstance(model, DensePhrases) or isinstance(model, SimCSE):
        model.model.eval()
    else:
        model.eval()
    with torch.no_grad():
        for i in range(0, len(context_list), batch_size):
            batch_text_list = text_list[i:i + batch_size]
            batch_context_list = context_list[i:i + batch_size]
            if isinstance(model, DensePhrases):
                inputs = model.tokenizer(batch_context_list, padding=True,
                    truncation=True, return_tensors='pt', max_length=512).to(
                    device)
                batch_emb_list = model.model.embed_phrase(**inputs)[0]
            elif isinstance(model, SimCSE):
                inputs = model.tokenizer(batch_context_list, padding=True,
                    truncation=True, return_tensors='pt', max_length=512).to(
                    device)
                outputs = model.model(**inputs, output_hidden_states=True,
                    return_dict=True)
                batch_emb_list = outputs.last_hidden_state
            else:
                batch_emb_list = model.encode(batch_context_list,
                    batch_size=batch_size, convert_to_tensor=True,
                    show_progress_bar=False, output_value='token_embeddings')
            contextual_phrase_embs = extract_contextual_phrase_embeddings(model
                , batch_text_list, batch_context_list, batch_emb_list)
            all_contextual_emb_tensor_list.extend(contextual_phrase_embs)
    return all_contextual_emb_tensor_list
