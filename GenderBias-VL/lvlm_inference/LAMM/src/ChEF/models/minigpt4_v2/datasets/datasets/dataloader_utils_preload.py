def preload(self, it):
    try:
        self.batch = next(it)
    except StopIteration:
        self.batch = None
        return
    with torch.cuda.stream(self.stream):
        self.batch = move_to_cuda(self.batch)
