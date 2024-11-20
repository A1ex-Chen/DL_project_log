def embedding_batch(self, list_inputText, max_length=64, contextual=False):
    inputs = self.tokenizer(list_inputText, truncation=True, return_tensors
        ='pt', max_length=max_length, padding='max_length')
    inputs = inputs.to(self.device)
    embeddings = self.model.embed_phrase(**inputs)[0]
    if contextual:
        return embeddings
    final_results = []
    for idx, token_embeddings in enumerate(embeddings):
        att_mask = list(inputs['attention_mask'][idx])
        last_token_index = att_mask.index(0) - 1 if 0 in att_mask else len(
            att_mask) - 1
        final_results.append(token_embeddings[:last_token_index + 1].mean(
            dim=0))
    return torch.stack(final_results)
