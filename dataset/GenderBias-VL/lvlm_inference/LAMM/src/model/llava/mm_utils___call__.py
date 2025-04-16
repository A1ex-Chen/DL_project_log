def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor,
    **kwargs) ->bool:
    outputs = []
    for i in range(output_ids.shape[0]):
        outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
    return all(outputs)
