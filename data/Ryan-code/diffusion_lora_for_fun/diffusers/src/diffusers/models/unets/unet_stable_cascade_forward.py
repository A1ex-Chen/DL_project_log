def forward(self, sample, timestep_ratio, clip_text_pooled, clip_text=None,
    clip_img=None, effnet=None, pixels=None, sca=None, crp=None,
    return_dict=True):
    if pixels is None:
        pixels = sample.new_zeros(sample.size(0), 3, 8, 8)
    timestep_ratio_embed = self.get_timestep_ratio_embedding(timestep_ratio)
    for c in self.config.timestep_conditioning_type:
        if c == 'sca':
            cond = sca
        elif c == 'crp':
            cond = crp
        else:
            cond = None
        t_cond = cond or torch.zeros_like(timestep_ratio)
        timestep_ratio_embed = torch.cat([timestep_ratio_embed, self.
            get_timestep_ratio_embedding(t_cond)], dim=1)
    clip = self.get_clip_embeddings(clip_txt_pooled=clip_text_pooled,
        clip_txt=clip_text, clip_img=clip_img)
    x = self.embedding(sample)
    if hasattr(self, 'effnet_mapper') and effnet is not None:
        x = x + self.effnet_mapper(nn.functional.interpolate(effnet, size=x
            .shape[-2:], mode='bilinear', align_corners=True))
    if hasattr(self, 'pixels_mapper'):
        x = x + nn.functional.interpolate(self.pixels_mapper(pixels), size=
            x.shape[-2:], mode='bilinear', align_corners=True)
    level_outputs = self._down_encode(x, timestep_ratio_embed, clip)
    x = self._up_decode(level_outputs, timestep_ratio_embed, clip)
    sample = self.clf(x)
    if not return_dict:
        return sample,
    return StableCascadeUNetOutput(sample=sample)
