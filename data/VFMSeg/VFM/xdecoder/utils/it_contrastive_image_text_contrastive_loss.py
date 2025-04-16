def image_text_contrastive_loss(image_feat, text_feat, temperature,
    image_id=None, text_id=None):
    image_feat = all_gather_grad(image_feat)
    text_feat = all_gather_grad(text_feat)
    logits = torch.matmul(image_feat, text_feat.t())
    logits /= temperature
    if image_id is None and text_id is None:
        gt = torch.arange(logits.shape[0], device=logits.device)
        loss1 = F.cross_entropy(logits, gt)
        loss2 = F.cross_entropy(logits.t(), gt)
    else:
        image_id = all_gather_grad(image_id)
        text_id = all_gather_grad(text_id)
        gt_image = image_id.reshape((-1, 1)) == image_id.reshape((1, -1))
        gt_text = text_id.reshape((-1, 1)) == text_id.reshape((1, -1))
        gt = torch.logical_or(gt_image, gt_text)
        loss1 = -torch.sum(gt * F.log_softmax(logits, dim=1)) / gt.sum()
        loss2 = -torch.sum(gt.t() * F.log_softmax(logits.t(), dim=1)) / gt.sum(
            )
    return (loss1 + loss2) / 2 * get_world_size()
