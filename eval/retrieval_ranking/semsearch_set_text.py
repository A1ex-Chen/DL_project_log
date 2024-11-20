def set_text(self, context, contextual=False, scorer=None, max_seq_length=128):
    sentences = tokenizer.tokenize(context)
    self.sentences.clear()
    self.contextual = contextual
    self.max_seq_length = int(max_seq_length)
    if scorer != 'USE':
        for sent in sentences:
            tokens = [token.text.lower() for token in self.nlp.tokenizer(sent)]
            encoded_sent = self.scorer.tokenizer.encode_plus(text=' '.join(
                tokens), add_special_tokens=True)
            encoded_sent = np.array(encoded_sent['input_ids'])[np.array(
                encoded_sent['attention_mask']) == 1]
            if len(encoded_sent) >= int(self.max_seq_length):
                continue
            self.sentences.append(sent)
    else:
        self.sentences = sentences
    if not contextual:
        self._update_phrases()
    else:
        self._update_phrases_with_index()
