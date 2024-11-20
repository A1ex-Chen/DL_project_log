@property
def steps_per_epoch(self):
    return self.cfg.SOLVER.NUM_IMAGES // self.train_batch_size
