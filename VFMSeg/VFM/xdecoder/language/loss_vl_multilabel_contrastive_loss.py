def vl_multilabel_contrastive_loss(image_feat, text_feat, temperature=1):
    """
    Args:
        image_feat (torch.Tensor): shape [B, L1, C] # B: batch_size, L1: 1, C: 256
        text_feat (torch.Tensor): shape [B, L2, C] # B:batch_size, L2: number of selected nouns, C: 256

    Returns:
    """
    dist_per_img = image_feat @ rearrange(text_feat, 'b l c -> b c l')
    dist_per_text = text_feat @ rearrange(image_feat, 'b l c -> b c l')
    batch = image_feat.shape[0]
    img_len = image_feat.shape[1]
    text_len = text_feat.shape[1]
    pos_labels_batch_img = rearrange(torch.ones_like(dist_per_text) /
        dist_per_text.size(1), 'b l2 l1 -> b l1 l2')
    pos_labels_batch_text = rearrange(torch.ones_like(dist_per_img) /
        dist_per_img.size(1), 'b l1 l2 -> b l2 l1')
    image_x = rearrange(image_feat, 'b l c -> (b l) c')
    text_x = rearrange(text_feat, 'b l c -> (b l) c')
    logits_per_img = image_x @ all_gather_grad(text_x).t()
    logits_per_text = text_x @ all_gather_grad(image_x).t()
    labels_per_img = F.one_hot(torch.ones(batch, img_len, batch, text_len,
        dtype=torch.long, device=image_x.device) * get_rank(), num_classes=
        get_world_size()).to(image_x.dtype)
    labels_per_img *= rearrange(pos_labels_batch_img, 'b l1 l2 -> b l1 1 l2 1'
        ) * repeat(torch.eye(batch, dtype=image_x.dtype, device=image_x.
        device), 'b1 b2 -> b1 1 b2 1 1')
    labels_per_img = rearrange(labels_per_img,
        'b1 l1 b2 l2 w -> (b1 l1) (w b2 l2)')
    labels_per_text = F.one_hot(torch.ones(batch, text_len, batch, img_len,
        dtype=torch.long, device=text_x.device) * get_rank(), num_classes=
        get_world_size()).to(text_x.dtype)
    labels_per_text *= rearrange(pos_labels_batch_text,
        'b l2 l1 -> b l2 1 l1 1') * repeat(torch.eye(batch, dtype=text_x.
        dtype, device=image_x.device), 'b2 b1 -> b2 1 b1 1 1')
    labels_per_text = rearrange(labels_per_text,
        'b2 l2 b1 l1 w -> (b2 l2) (w b1 l1)')
    logit_scale = temperature.exp().clamp(max=100)
    loss_img = soft_cross_entropy(logit_scale * logits_per_img, labels_per_img)
    loss_text = soft_cross_entropy(logit_scale * logits_per_text,
        labels_per_text)
    loss = 0.5 * (loss_img + loss_text)
    return loss
