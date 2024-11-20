def prefetch(self):
    with torch.cuda.stream(self.prefetch_stream):
        try:
            self.prefetched_data = self.fetch_next()
        except StopIteration:
            self.prefetched_data = None
