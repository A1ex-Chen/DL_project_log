def progress_bar(self, iterable=None, total=None):
    self.prior_pipe.progress_bar(iterable=iterable, total=total)
    self.decoder_pipe.progress_bar(iterable=iterable, total=total)
    self.decoder_pipe.enable_model_cpu_offload()
