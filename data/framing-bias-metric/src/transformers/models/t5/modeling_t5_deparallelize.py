@add_start_docstrings(DEPARALLELIZE_DOCSTRING)
def deparallelize(self):
    self.encoder.deparallelize()
    self.decoder.deparallelize()
    self.encoder = self.encoder.to('cpu')
    self.decoder = self.decoder.to('cpu')
    self.lm_head = self.lm_head.to('cpu')
    self.model_parallel = False
    self.device_map = None
    torch.cuda.empty_cache()
