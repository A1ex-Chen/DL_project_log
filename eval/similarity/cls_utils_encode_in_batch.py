def encode_in_batch(model, batch_size, text_list, device):
    all_emb_tensor_list = []
    if isinstance(model, DensePhrases) or isinstance(model, SimCSE):
        model.model.eval()
    elif isinstance(model, SentenceTransformer):
        model.eval()
    with torch.no_grad():
        for i in range(0, len(text_list), batch_size):
            batch_text_list = text_list[i:i + batch_size]
            if isinstance(model, DensePhrases):
                inputs = model.tokenizer(batch_text_list, padding=True,
                    truncation=True, return_tensors='pt', max_length=128).to(
                    device)
                embeddings = model.model.embed_phrase(**inputs)[0]
                batch_emb_list = []
                for emb_idx, token_embeddings in enumerate(embeddings):
                    att_mask = list(inputs['attention_mask'][emb_idx])
                    last_token_index = att_mask.index(0
                        ) - 1 if 0 in att_mask else len(att_mask) - 1
                    batch_emb_list.append(token_embeddings[:
                        last_token_index + 1].mean(dim=0))
            elif isinstance(model, SimCSE):
                inputs = model.tokenizer(batch_text_list, padding=True,
                    truncation=True, return_tensors='pt', max_length=128).to(
                    device)
                outputs = model.model(**inputs, output_hidden_states=True,
                    return_dict=True)
                batch_emb_list = []
                for idx, token_embeddings in enumerate(outputs.
                    last_hidden_state):
                    att_mask = list(inputs['attention_mask'][idx])
                    last_token_index = att_mask.index(0
                        ) - 1 if 0 in att_mask else len(att_mask) - 1
                    batch_emb_list.append(token_embeddings[:
                        last_token_index + 1].mean(dim=0))
            elif isinstance(model, SentenceTransformer):
                batch_emb_list = model.encode(batch_text_list, batch_size=
                    batch_size, convert_to_tensor=True, show_progress_bar=False
                    )
            all_emb_tensor_list.extend(batch_emb_list)
    return all_emb_tensor_list
