def forward(self, input: torch.Tensor) ->torch.Tensor:
    if torch.jit.is_scripting():
        bias = self.bias.to(dtype=input.dtype
            ) if self.bias is not None else None
        return F.linear(input, self.weight.to(dtype=input.dtype), bias=bias)
    else:
        return F.linear(input, self.weight, self.bias)
