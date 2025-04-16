def init_mems(self):
    if self.mem_len > 0:
        param = next(self.parameters())
        mems = torch.empty(self.n_layer, 0, dtype=param.dtype, device=param
            .device)
        return mems
    else:
        return None
