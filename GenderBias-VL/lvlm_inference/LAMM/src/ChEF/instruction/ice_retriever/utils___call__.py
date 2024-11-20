def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) ->BatchEncoding:
    res_dict = ignore_pad_dict(features)
    has_labels = 'labels' in features[0]
    if has_labels:
        labels = [{'input_ids': x.pop('labels')} for x in features]
        labels = self.tokenizer.pad(labels, padding=True, max_length=self.
            max_length, pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=True, return_tensors='pt', verbose=False)
    batch = self.tokenizer.pad(features, padding=True, max_length=self.
        max_length, pad_to_multiple_of=self.pad_to_multiple_of,
        return_attention_mask=True, return_tensors='pt', verbose=False)
    if has_labels:
        batch['labels'] = labels.input_ids
    batch.update(res_dict)
    if self.device:
        batch = batch.to(self.device)
    return batch
