def on_epoch_begin(self, epoch: int, num_batches_in_epoch: int):
    self.time_checkpoint = time.time()
    self.batches_counter = 0
