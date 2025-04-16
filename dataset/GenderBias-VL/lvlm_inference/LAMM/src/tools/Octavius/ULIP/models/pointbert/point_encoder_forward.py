def forward(self, pts):
    neighborhood, center = self.group_divider(pts)
    neighborhood = neighborhood[..., :3]
    group_input_tokens = self.encoder(neighborhood)
    group_input_tokens = self.reduce_dim(group_input_tokens)
    cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
    cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
    pos = self.pos_embed(center)
    x = torch.cat((cls_tokens, group_input_tokens), dim=1)
    pos = torch.cat((cls_pos, pos), dim=1)
    x = self.blocks(x, pos)
    x = self.norm(x)
    concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
    return concat_f
