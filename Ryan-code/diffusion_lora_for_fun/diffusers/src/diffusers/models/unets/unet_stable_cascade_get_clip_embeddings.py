def get_clip_embeddings(self, clip_txt_pooled, clip_txt=None, clip_img=None):
    if len(clip_txt_pooled.shape) == 2:
        clip_txt_pool = clip_txt_pooled.unsqueeze(1)
    clip_txt_pool = self.clip_txt_pooled_mapper(clip_txt_pooled).view(
        clip_txt_pooled.size(0), clip_txt_pooled.size(1) * self.config.
        clip_seq, -1)
    if clip_txt is not None and clip_img is not None:
        clip_txt = self.clip_txt_mapper(clip_txt)
        if len(clip_img.shape) == 2:
            clip_img = clip_img.unsqueeze(1)
        clip_img = self.clip_img_mapper(clip_img).view(clip_img.size(0), 
            clip_img.size(1) * self.config.clip_seq, -1)
        clip = torch.cat([clip_txt, clip_txt_pool, clip_img], dim=1)
    else:
        clip = clip_txt_pool
    return self.clip_norm(clip)
