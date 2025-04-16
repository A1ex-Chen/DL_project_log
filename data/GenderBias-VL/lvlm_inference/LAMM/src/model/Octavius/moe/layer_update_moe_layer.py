def update_moe_layer(self, adapter_name, r, num_experts, lora_alpha,
    lora_dropout, init_lora_weights):
    self.r[adapter_name] = r
    self.lora_alpha[adapter_name] = lora_alpha
    if lora_dropout > 0.0:
        lora_dropout_layer = nn.Dropout(p=lora_dropout)
    else:
        lora_dropout_layer = nn.Identity()
    self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
    if r > 0:
        lora_A = nn.Parameter(torch.zeros(num_experts, self.in_features, r))
        lora_B = nn.Parameter(torch.zeros(num_experts, r, self.out_features))
        self.lora_moe_A.update(nn.ParameterDict({adapter_name: lora_A}))
        self.lora_moe_B.update(nn.ParameterDict({adapter_name: lora_B}))
        self.scaling[adapter_name] = lora_alpha / r
    if init_lora_weights:
        self.reset_lora_parameters(adapter_name)
    self.to(self.weight.device)
