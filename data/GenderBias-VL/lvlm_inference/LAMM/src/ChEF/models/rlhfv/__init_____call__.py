def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor,
    **kwargs) ->bool:
    for o in output_ids:
        o = self.tokenizer.decode(o[self.input_size:], skip_special_tokens=True
            )
        if all([(keyword not in o) for keyword in self.keywords]):
            return False
    return True
