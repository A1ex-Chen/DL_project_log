def __call__(self, examples: List[Union[List[int], torch.Tensor, Dict[str,
    torch.Tensor]]]) ->Dict[str, torch.Tensor]:
    if isinstance(examples[0], (dict, BatchEncoding)):
        examples = [e['input_ids'] for e in examples]
    batch = _collate_batch(examples, self.tokenizer)
    inputs, perm_mask, target_mapping, labels = self.mask_tokens(batch)
    return {'input_ids': inputs, 'perm_mask': perm_mask, 'target_mapping':
        target_mapping, 'labels': labels}
