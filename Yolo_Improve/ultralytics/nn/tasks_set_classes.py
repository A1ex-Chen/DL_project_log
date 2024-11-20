def set_classes(self, text, batch=80, cache_clip_model=True):
    """Set classes in advance so that model could do offline-inference without clip model."""
    try:
        import clip
    except ImportError:
        check_requirements('git+https://github.com/ultralytics/CLIP.git')
        import clip
    if not getattr(self, 'clip_model', None) and cache_clip_model:
        self.clip_model = clip.load('ViT-B/32')[0]
    model = self.clip_model if cache_clip_model else clip.load('ViT-B/32')[0]
    device = next(model.parameters()).device
    text_token = clip.tokenize(text).to(device)
    txt_feats = [model.encode_text(token).detach() for token in text_token.
        split(batch)]
    txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats,
        dim=0)
    txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
    self.txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1])
    self.model[-1].nc = len(text)
