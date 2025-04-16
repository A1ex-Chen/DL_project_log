def torch_call(self, examples):
    if isinstance(examples[0], Mapping):
        batch = self.tokenizer.pad(examples, return_tensors='pt',
            pad_to_multiple_of=self.pad_to_multiple_of)
    else:
        batch = {'input_ids': _torch_collate_batch(examples, self.tokenizer,
            pad_to_multiple_of=self.pad_to_multiple_of)}
    if self.mlm:
        batch['input_ids'], batch['labels'] = self.my_torch_mask_tokens(batch
            ['input_ids'], batch['labels'])
    else:
        labels = batch['input_ids'].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        batch['labels'] = labels
    return batch
