@add_start_docstrings(DEPARALLELIZE_DOCSTRING)
def deparallelize(self):
    self.encoder.deparallelize()
    self.encoder = self.encoder.to('cpu')
    self.model_parallel = False
    self.device_map = None
    torch.cuda.empty_cache()
