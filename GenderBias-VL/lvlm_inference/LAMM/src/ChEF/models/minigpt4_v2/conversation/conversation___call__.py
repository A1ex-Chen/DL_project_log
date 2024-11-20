def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
    for stop in self.stops:
        if torch.all(input_ids[:, -len(stop):] == stop).item():
            return True
    return False
