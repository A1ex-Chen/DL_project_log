def next(self):
    torch.cuda.current_stream().wait_stream(self.stream)
    input = self.next_input
    target = self.next_target
    self.preload()
    return input, target
