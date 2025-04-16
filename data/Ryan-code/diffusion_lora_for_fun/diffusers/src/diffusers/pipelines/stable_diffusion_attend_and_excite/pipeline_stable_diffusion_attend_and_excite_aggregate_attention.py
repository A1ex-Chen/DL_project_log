def aggregate_attention(self, from_where: List[str]) ->torch.Tensor:
    """Aggregates the attention across the different layers and heads at the specified resolution."""
    out = []
    attention_maps = self.get_average_attention()
    for location in from_where:
        for item in attention_maps[location]:
            cross_maps = item.reshape(-1, self.attn_res[0], self.attn_res[1
                ], item.shape[-1])
            out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out
