def forward_visual(self, imgs):
    clip_feats = self.clip_encode_image(imgs)
    clip_feats = self.clip_proj_norm(self.clip_proj(clip_feats.float()))
    visual_query = self.visual_query.weight.unsqueeze(0).repeat(len(imgs), 1, 1
        )
    visual_query = torch.cat([visual_query, clip_feats], dim=1)
    for block in self.visual_blocks:
        visual_query = block(visual_query)
    visual_query = visual_query[:, :self.query_len, :]
    visual_query = self.visual_proj(visual_query)
    visual_query = self.visual_proj_norm(visual_query)
    return visual_query
