def encode_line(self, tokenizer, line, max_length, pad_to_max_length=True,
    return_tensors='pt'):
    """Only used by LegacyDataset"""
    return tokenizer([line], max_length=max_length, padding='max_length' if
        pad_to_max_length else None, truncation=True, return_tensors=
        return_tensors, **self.dataset_kwargs)
