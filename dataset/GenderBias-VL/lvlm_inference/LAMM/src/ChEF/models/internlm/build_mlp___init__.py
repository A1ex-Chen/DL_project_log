def __init__(self, in_features: int, out_features: int, bias: bool=True,
    device=None, dtype=None, lora_r=8, lora_alpha=16, lora_dropout=0.05,
    lora_len=0, **kwargs) ->None:
    super().__init__(in_features, out_features, bias, device, dtype)
    self.lora_r = lora_r
    self.lora_alpha = lora_alpha
    self.lora_len = lora_len
    if lora_dropout > 0.0:
        self.lora_dropout = nn.Dropout(p=lora_dropout)
    else:
        self.lora_dropout = lambda x: x
    self.lora_scaling = self.lora_alpha / self.lora_r
    self.Plora_A = nn.Linear(in_features, self.lora_r, bias=False, device=
        device, dtype=dtype)
    self.Plora_B = nn.Linear(self.lora_r, out_features, bias=False, device=
        device, dtype=dtype)
    self.reset_parameters()
