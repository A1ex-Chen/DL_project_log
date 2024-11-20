def get_clip_text_emb(self, texts):
    with torch.no_grad():
        text = clip.tokenize(texts).to(self.device)
        text_features = self.clip_model.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features
