def to(self, torch_device: Optional[Union[str, torch.device]]=None,
    torch_dtype: Optional[torch.dtype]=None):
    self.mean = nn.Parameter(self.mean.to(torch_device).to(torch_dtype))
    self.std = nn.Parameter(self.std.to(torch_device).to(torch_dtype))
    return self
