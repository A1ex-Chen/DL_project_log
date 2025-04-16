def vl_similarity(image_feat, text_feat, temperature=1):
    logits = torch.matmul(image_feat, text_feat.t())
    logits = temperature.exp().clamp(max=100) * logits
    return logits
