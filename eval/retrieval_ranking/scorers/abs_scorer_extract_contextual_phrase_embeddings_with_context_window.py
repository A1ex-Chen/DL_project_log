def extract_contextual_phrase_embeddings_with_context_window(self,
    list_phrase, context_phrases, context_phrase_embs, max_length=256):
    all_phrase_embs = []
    encoded_phrase_list = self.tokenizer.batch_encode_plus(list_phrase,
        max_length=128, padding='max_length', truncation=True,
        add_special_tokens=True)
    encoded_context_list = self.tokenizer.batch_encode_plus(context_phrases,
        max_length=max_length, padding='max_length', truncation=True,
        add_special_tokens=True)
    for idx, (encoded_phrase, encoded_sent) in enumerate(zip(
        encoded_phrase_list['input_ids'], encoded_context_list['input_ids'])):
        encoded_phrase = np.array(encoded_phrase)[np.array(
            encoded_phrase_list['attention_mask'][idx]) == 1]
        encoded_sent = np.array(encoded_sent)[np.array(encoded_context_list
            ['attention_mask'][idx]) == 1]
        start_idx, end_idx = self.find_sub_list(list(encoded_phrase[1:-1]),
            list(encoded_sent))
        phrase_indices = list(range(start_idx, end_idx + 1, 1))
        phrase_embs = context_phrase_embs[idx][phrase_indices].mean(dim=0)
        if all(np.isnan(phrase_embs.numpy())):
            continue
        all_phrase_embs.append(phrase_embs)
    return torch.stack(all_phrase_embs)
