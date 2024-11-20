def set_model_attributes(self):
    """Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
    self.model.nc = self.data['nc']
    self.model.names = self.data['names']
    self.model.args = self.args
