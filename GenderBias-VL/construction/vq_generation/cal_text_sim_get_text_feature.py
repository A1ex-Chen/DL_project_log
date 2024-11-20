def get_text_feature(text, clip_model):
    text = clip.tokenize(text).to('cuda')
    text_features = clip_model.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features
