@torch.jit.ignore
def _forward(self, x: torch.Tensor) ->torch.Tensor:
    x = x + sum(attn(x) for attn in self.attns)
    x = x + sum(ffn(x) for ffn in self.ffns)
    return x
