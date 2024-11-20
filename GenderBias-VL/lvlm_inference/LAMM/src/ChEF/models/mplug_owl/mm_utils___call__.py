def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor,
    **kwargs) ->bool:
    assert output_ids.shape[0] == 1, 'Only support batch size 1 (yet)'
    offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
    self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in
        self.keyword_ids]
    for keyword_id in self.keyword_ids:
        if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
            return True
    outputs = self.tokenizer.batch_decode(output_ids[:, -offset:],
        skip_special_tokens=True)[0]
    for keyword in self.keywords:
        if keyword in outputs:
            return True
    return False
