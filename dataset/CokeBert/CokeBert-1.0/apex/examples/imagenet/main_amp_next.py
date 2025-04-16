def next(self):
    torch.cuda.current_stream().wait_stream(self.stream)
    input = self.next_input
    target = self.next_target
    if input is not None:
        input.record_stream(torch.cuda.current_stream())
    if target is not None:
        target.record_stream(torch.cuda.current_stream())
    self.preload()
    return input, target
