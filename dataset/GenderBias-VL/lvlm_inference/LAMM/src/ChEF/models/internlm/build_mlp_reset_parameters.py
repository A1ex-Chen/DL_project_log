def reset_parameters(self):
    if hasattr(self, 'lora_A'):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
