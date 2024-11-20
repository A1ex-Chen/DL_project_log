def forward(self, input: Tensor, unembed: bool=False) ->Tensor:
    if unembed:
        return F.linear(input, self.weight)
    return super().forward(input)
