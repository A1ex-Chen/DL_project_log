def tokenize_input_and_cls_pos(self, input, stem):
    tokens = stem.norm_layer(stem.proj(input))
    assert tokens.ndim == 3
    assert tokens.shape[2] == self.embed_dim
    B = tokens.shape[0]
    if self.num_cls_tokens > 0:
        class_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat((class_tokens, tokens), dim=1)
    if self.use_pos_embed:
        tokens = tokens + self.pos_embed
    return tokens
