def image_text_contrastive_loss_queue(image_feat_inp, text_feat_inp,
    lang_enc, training):
    image_feat = all_gather_grad(image_feat_inp.contiguous())
    text_feat = all_gather_grad(text_feat_inp.contiguous())
    image_feat = image_feat / (image_feat.norm(dim=-1, keepdim=True) + 1e-07)
    text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-07)
    temperature = lang_enc.logit_scale
    logits = torch.matmul(image_feat, text_feat.t())
    logit_scale = temperature.exp().clamp(max=100)
    gt = torch.arange(logits.shape[0], device=logits.device)
    loss1 = F.cross_entropy(logit_scale * logits, gt)
    loss2 = F.cross_entropy(logit_scale * logits.t(), gt)
    return (loss1 + loss2) / 2
