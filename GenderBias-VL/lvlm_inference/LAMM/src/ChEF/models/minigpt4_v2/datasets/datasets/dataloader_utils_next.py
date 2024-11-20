def next(self, it):
    torch.cuda.current_stream().wait_stream(self.stream)
    batch = self.batch
    if batch is not None:
        record_cuda_stream(batch)
    self.preload(it)
    return batch
