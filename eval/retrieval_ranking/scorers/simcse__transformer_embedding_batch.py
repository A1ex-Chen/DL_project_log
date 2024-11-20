def _transformer_embedding_batch(self, list_inputText, max_length=64,
    contextual=False):
    inputs = self.tokenizer(list_inputText, max_length=max_length, padding=
        'max_length', truncation=True, add_special_tokens=True,
        return_tensors='pt')
    inputs = inputs.to(self.device)
    outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
    if contextual:
        return outputs.last_hidden_state
    final_results = []
    for idx, token_embeddings in enumerate(outputs.last_hidden_state):
        att_mask = list(inputs['attention_mask'][idx])
        last_token_index = att_mask.index(0) - 1 if 0 in att_mask else len(
            att_mask) - 1
        final_results.append(token_embeddings[:last_token_index + 1].mean(
            dim=0))
    return torch.stack(final_results)
