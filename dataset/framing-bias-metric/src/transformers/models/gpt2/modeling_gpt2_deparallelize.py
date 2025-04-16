@add_start_docstrings(DEPARALLELIZE_DOCSTRING)
def deparallelize(self):
    self.transformer.deparallelize()
    self.transformer = self.transformer.to('cpu')
    self.lm_head = self.lm_head.to('cpu')
    self.model_parallel = False
    torch.cuda.empty_cache()
