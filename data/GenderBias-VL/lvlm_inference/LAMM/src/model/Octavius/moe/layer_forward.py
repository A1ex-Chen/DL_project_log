def forward(self, x: torch.Tensor):
    previous_dtype = x.dtype
    if self.active_adapter not in self.lora_A.keys(
        ) and self.active_adapter not in self.lora_moe_A.keys():
        return F.linear(x, transpose(self.weight, self.fan_in_fan_out),
            bias=self.bias)
    if self.disable_adapters:
        if self.r[self.active_adapter] > 0 and self.merged:
            self.unmerge()
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out),
            bias=self.bias)
    elif self.r[self.active_adapter] > 0 and not self.merged:
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out),
            bias=self.bias)
        x = x.to(self.lora_moe_A[self.active_adapter].dtype)
        out = self.lora_dropout[self.active_adapter](x)
        out = torch.einsum('bnd,edh->bneh', out, self.lora_moe_A[self.
            active_adapter])
        out = torch.einsum('bneh,ehd->bned', out, self.lora_moe_B[self.
            active_adapter])
        if self.gate_mode == 'top2_gate':
            assert self.moe_gate is not None
            soft_gate = self.moe_gate
            out = torch.einsum('bned,be->bned', out, soft_gate)
        out = out.sum(dim=2)
        result += out
    else:
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out),
            bias=self.bias)
    result = result.to(previous_dtype)
    return result
