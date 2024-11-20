def forward_head(self, x: torch.Tensor, pre_logits: bool=False) ->torch.Tensor:
    if self.attn_pool is not None:
        x = self.attn_pool(x)
    elif self.global_pool == 'avg':
        x = x[:, self.num_prefix_tokens:].mean(dim=1)
    elif self.global_pool:
        x = x[:, 0]
    x = self.fc_norm(x)
    x = self.head_drop(x)
    return x if pre_logits else self.head(x)
