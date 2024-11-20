def ql_multi_contrastive_loss(image_feat, text_feat, text_hash, temperature=1):
    image_feat = all_gather_arbitary_tensor(image_feat)
    text_feat = all_gather_arbitary_tensor(text_feat)
    text_hash_batch = all_gather_pickle(text_hash, text_feat.device)
    text_hash_all = torch.cat(text_hash_batch)
    text_hash_all_unique = torch.unique(text_hash_all).tolist()
    gt = torch.zeros((image_feat.shape[0], len(text_hash_all_unique)),
        device=text_feat.device)
    text_hash_all = text_hash_all.tolist()
    text_feat_unique = torch.stack([text_feat[text_hash_all.index(txt)] for
        txt in text_hash_all_unique])
    for idx, txt in enumerate(text_hash_all):
        gt[idx][text_hash_all_unique.index(txt)] = 1
    logits = torch.matmul(image_feat, text_feat_unique.t())
    logits = logits * temperature.exp().clamp(max=100)
    loss_img = soft_cross_entropy(logits, gt)
    loss_text = soft_cross_entropy(logits.t(), gt.t() / gt.t().sum(-1,
        keepdim=True))
    loss = 0.7 * loss_img + 0.3 * loss_text
    return loss
