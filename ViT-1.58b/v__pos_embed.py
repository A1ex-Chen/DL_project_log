def _pos_embed(self, x: torch.Tensor) ->torch.Tensor:
    if self.dynamic_img_size:
        B, H, W, C = x.shape
        pos_embed = resample_abs_pos_embed(self.pos_embed, (H, W),
            num_prefix_tokens=0 if self.no_embed_class else self.
            num_prefix_tokens)
        x = x.view(B, -1, C)
    else:
        pos_embed = self.pos_embed
    to_cat = []
    if self.cls_token is not None:
        to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
    if self.reg_token is not None:
        to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))
    if self.no_embed_class:
        x = x + pos_embed
        if to_cat:
            x = torch.cat(to_cat + [x], dim=1)
    else:
        if to_cat:
            x = torch.cat(to_cat + [x], dim=1)
        x = x + pos_embed
    return self.pos_drop(x)
