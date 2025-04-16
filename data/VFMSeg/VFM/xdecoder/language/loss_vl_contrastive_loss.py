def vl_contrastive_loss(image_feat, text_feat, temperature=1):
    image_feat = all_gather_grad(image_feat)
    text_feat = all_gather_grad(text_feat)
    logits = torch.matmul(image_feat, text_feat.t())
    logit_scale = temperature.exp().clamp(max=100)
    gt = torch.arange(logits.shape[0], device=logits.device)
    loss1 = F.cross_entropy(logit_scale * logits, gt)
    loss2 = F.cross_entropy(logit_scale * logits.t(), gt)
    return (loss1 + loss2) / 2
