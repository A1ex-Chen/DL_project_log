def init_mems(self):
    mems = torch.empty(self.n_layer, 0, dtype=self.dtype, device=torch.
        device('cuda'))
    return mems
