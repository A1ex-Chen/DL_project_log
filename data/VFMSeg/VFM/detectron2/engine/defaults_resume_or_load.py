def resume_or_load(self, resume=True):
    """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.

        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.

        Args:
            resume (bool): whether to do resume or not
        """
    self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
    if resume and self.checkpointer.has_checkpoint():
        self.start_iter = self.iter + 1
