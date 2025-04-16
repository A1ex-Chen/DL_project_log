def on_val_batch_end(self, batch_i, im, targets, paths, shapes, out):
    if self.comet_logger:
        self.comet_logger.on_val_batch_end(batch_i, im, targets, paths,
            shapes, out)
