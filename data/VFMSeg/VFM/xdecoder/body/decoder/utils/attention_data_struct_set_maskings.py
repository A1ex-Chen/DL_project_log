def set_maskings(self, name, masking):
    self.attn_variables[name].masking = masking
