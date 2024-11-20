def reset_queue_ptr(self):
    self.queue_ptr = torch.zeros(1, dtype=torch.long)
