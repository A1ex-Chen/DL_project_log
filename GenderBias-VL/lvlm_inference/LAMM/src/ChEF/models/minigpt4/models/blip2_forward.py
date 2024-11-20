def forward(self, x: torch.Tensor):
    orig_type = x.dtype
    ret = super().forward(x.type(torch.float32))
    return ret.type(orig_type)
