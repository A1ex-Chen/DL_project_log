def _forward_jit(self, x: torch.Tensor) ->torch.Tensor:
    x = x + torch.stack([attn(x) for attn in self.attns]).sum(dim=0)
    x = x + torch.stack([ffn(x) for ffn in self.ffns]).sum(dim=0)
    return x
