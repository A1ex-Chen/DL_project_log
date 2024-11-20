def forward(self, vision_embeds_3d, mask):
    vision_embeds = self.query_embed.weight.unsqueeze(0).repeat(
        vision_embeds_3d.shape[0], 1, 1)
    for i in range(self.num_layers):
        vision_embeds = self.cross_attn[i](vision_embeds, vision_embeds_3d,
            vision_embeds_3d, attention_mask=mask)
        vision_embeds = self.self_attn[i](vision_embeds, vision_embeds,
            vision_embeds)
    return vision_embeds
