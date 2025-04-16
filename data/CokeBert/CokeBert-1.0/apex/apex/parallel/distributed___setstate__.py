def __setstate__(self, state):
    super(DistributedDataParallel, self).__setstate__(state)
    self.reduction_stream = torch.cuda.Stream()
    self.reduction_event = torch.cuda.Event(enable_timing=False, blocking=False
        )
