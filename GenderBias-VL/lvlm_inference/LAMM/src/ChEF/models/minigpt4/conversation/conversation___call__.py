def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
    for x in input_ids:
        end_now = False
        for stop in self.stops:
            stop = stop.to(x.device)
            end_now |= self._contains_subsequence(x[self.prompt_len:], stop)
        if not end_now:
            return False
    return True
