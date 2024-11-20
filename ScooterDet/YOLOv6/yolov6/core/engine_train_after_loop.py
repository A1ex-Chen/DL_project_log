def train_after_loop(self):
    if self.device != 'cpu':
        torch.cuda.empty_cache()
