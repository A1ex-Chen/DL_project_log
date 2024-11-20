def __init__(self, adapter_name: str, in_features: int, out_features: int,
    r: int=0, num_experts: int=16, gate_mode: str='top2_gate', lora_alpha:
    int=1, lora_dropout: float=0.0, fan_in_fan_out: bool=False, **kwargs):
    init_lora_weights = kwargs.pop('init_lora_weights', True)
    nn.Linear.__init__(self, in_features, out_features, **kwargs)
    MoeLoraLayer.__init__(self, in_features=in_features, out_features=
        out_features)
    self.weight.requires_grad = False
    self.fan_in_fan_out = fan_in_fan_out
    if fan_in_fan_out:
        self.weight.data = self.weight.data.T
    nn.Linear.reset_parameters(self)
    self.active_adapter = adapter_name
    self.num_experts = num_experts
    self.gate_mode = gate_mode
    self.moe_gate = None
    assert self.gate_mode in ['top2_gate']
    self.update_moe_top2_layer(adapter_name, r, num_experts, lora_alpha,
        lora_dropout, init_lora_weights)
