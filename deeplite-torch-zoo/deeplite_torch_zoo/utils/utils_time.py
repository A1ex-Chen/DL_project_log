def time(self):
    if self.cuda:
        torch.cuda.synchronize()
    return time.time()
