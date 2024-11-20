def set_progress_bar_config(self, **kwargs):
    self.prior_pipe.set_progress_bar_config(**kwargs)
    self.decoder_pipe.set_progress_bar_config(**kwargs)
