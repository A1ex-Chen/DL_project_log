def forward(self, inputs):
    if hasattr(self, 'rbr_reparam'):
        return self.act(self.rbr_reparam(inputs))
    if self.rbr_identity is None:
        id_out = 0
    else:
        id_out = self.rbr_identity(inputs)
    return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)
