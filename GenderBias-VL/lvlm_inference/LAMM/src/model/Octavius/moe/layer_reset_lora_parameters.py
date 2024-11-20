def reset_lora_parameters(self, adapter_name):
    if adapter_name in self.lora_moe_A.keys():
        nn.init.kaiming_uniform_(self.lora_moe_A[adapter_name], a=math.sqrt(5))
        nn.init.zeros_(self.lora_moe_B[adapter_name])
    if adapter_name in self.lora_moe_A.keys():
        nn.init.kaiming_uniform_(self.lora_moe_A[adapter_name], a=math.sqrt(5))
        nn.init.zeros_(self.lora_moe_B[adapter_name])
