def __call__(self, batch) ->Dict[str, torch.Tensor]:
    if hasattr(self.tokenizer, 'prepare_seq2seq_batch'):
        batch = self._encode(batch)
        input_ids, attention_mask, labels = batch['input_ids'], batch[
            'attention_mask'], batch['labels']
    else:
        input_ids = torch.stack([x['input_ids'] for x in batch])
        attention_mask = torch.stack([x['attention_mask'] for x in batch])
        labels = torch.stack([x['labels'] for x in batch])
        labels = trim_batch(labels, self.pad_token_id)
        input_ids, attention_mask = trim_batch(input_ids, self.pad_token_id,
            attention_mask=attention_mask)
    if isinstance(self.tokenizer, T5Tokenizer):
        decoder_input_ids = self._shift_right_t5(labels)
    else:
        decoder_input_ids = shift_tokens_right(labels, self.pad_token_id)
    batch = {'input_ids': input_ids, 'attention_mask': attention_mask,
        'decoder_input_ids': decoder_input_ids, 'labels': labels}
    return batch
