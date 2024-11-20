def forward(self, x: torch.Tensor, conditioning_emb: torch.Tensor
    ) ->torch.Tensor:
    emb = self.scale_bias(conditioning_emb)
    scale, shift = torch.chunk(emb, 2, -1)
    x = x * (1 + scale) + shift
    return x
