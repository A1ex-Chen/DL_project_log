def get_cdn_group(batch, num_classes, num_queries, class_embed, num_dn=100,
    cls_noise_ratio=0.5, box_noise_scale=1.0, training=False):
    """
    Get contrastive denoising training group. This function creates a contrastive denoising training group with positive
    and negative samples from the ground truths (gt). It applies noise to the class labels and bounding box coordinates,
    and returns the modified labels, bounding boxes, attention mask and meta information.

    Args:
        batch (dict): A dict that includes 'gt_cls' (torch.Tensor with shape [num_gts, ]), 'gt_bboxes'
            (torch.Tensor with shape [num_gts, 4]), 'gt_groups' (List(int)) which is a list of batch size length
            indicating the number of gts of each image.
        num_classes (int): Number of classes.
        num_queries (int): Number of queries.
        class_embed (torch.Tensor): Embedding weights to map class labels to embedding space.
        num_dn (int, optional): Number of denoising. Defaults to 100.
        cls_noise_ratio (float, optional): Noise ratio for class labels. Defaults to 0.5.
        box_noise_scale (float, optional): Noise scale for bounding box coordinates. Defaults to 1.0.
        training (bool, optional): If it's in training mode. Defaults to False.

    Returns:
        (Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Dict]]): The modified class embeddings,
            bounding boxes, attention mask and meta information for denoising. If not in training mode or 'num_dn'
            is less than or equal to 0, the function returns None for all elements in the tuple.
    """
    if not training or num_dn <= 0:
        return None, None, None, None
    gt_groups = batch['gt_groups']
    total_num = sum(gt_groups)
    max_nums = max(gt_groups)
    if max_nums == 0:
        return None, None, None, None
    num_group = num_dn // max_nums
    num_group = 1 if num_group == 0 else num_group
    bs = len(gt_groups)
    gt_cls = batch['cls']
    gt_bbox = batch['bboxes']
    b_idx = batch['batch_idx']
    dn_cls = gt_cls.repeat(2 * num_group)
    dn_bbox = gt_bbox.repeat(2 * num_group, 1)
    dn_b_idx = b_idx.repeat(2 * num_group).view(-1)
    neg_idx = torch.arange(total_num * num_group, dtype=torch.long, device=
        gt_bbox.device) + num_group * total_num
    if cls_noise_ratio > 0:
        mask = torch.rand(dn_cls.shape) < cls_noise_ratio * 0.5
        idx = torch.nonzero(mask).squeeze(-1)
        new_label = torch.randint_like(idx, 0, num_classes, dtype=dn_cls.
            dtype, device=dn_cls.device)
        dn_cls[idx] = new_label
    if box_noise_scale > 0:
        known_bbox = xywh2xyxy(dn_bbox)
        diff = (dn_bbox[..., 2:] * 0.5).repeat(1, 2) * box_noise_scale
        rand_sign = torch.randint_like(dn_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(dn_bbox)
        rand_part[neg_idx] += 1.0
        rand_part *= rand_sign
        known_bbox += rand_part * diff
        known_bbox.clip_(min=0.0, max=1.0)
        dn_bbox = xyxy2xywh(known_bbox)
        dn_bbox = torch.logit(dn_bbox, eps=1e-06)
    num_dn = int(max_nums * 2 * num_group)
    dn_cls_embed = class_embed[dn_cls]
    padding_cls = torch.zeros(bs, num_dn, dn_cls_embed.shape[-1], device=
        gt_cls.device)
    padding_bbox = torch.zeros(bs, num_dn, 4, device=gt_bbox.device)
    map_indices = torch.cat([torch.tensor(range(num), dtype=torch.long) for
        num in gt_groups])
    pos_idx = torch.stack([(map_indices + max_nums * i) for i in range(
        num_group)], dim=0)
    map_indices = torch.cat([(map_indices + max_nums * i) for i in range(2 *
        num_group)])
    padding_cls[dn_b_idx, map_indices] = dn_cls_embed
    padding_bbox[dn_b_idx, map_indices] = dn_bbox
    tgt_size = num_dn + num_queries
    attn_mask = torch.zeros([tgt_size, tgt_size], dtype=torch.bool)
    attn_mask[num_dn:, :num_dn] = True
    for i in range(num_group):
        if i == 0:
            attn_mask[max_nums * 2 * i:max_nums * 2 * (i + 1), max_nums * 2 *
                (i + 1):num_dn] = True
        if i == num_group - 1:
            attn_mask[max_nums * 2 * i:max_nums * 2 * (i + 1), :max_nums *
                i * 2] = True
        else:
            attn_mask[max_nums * 2 * i:max_nums * 2 * (i + 1), max_nums * 2 *
                (i + 1):num_dn] = True
            attn_mask[max_nums * 2 * i:max_nums * 2 * (i + 1), :max_nums * 
                2 * i] = True
    dn_meta = {'dn_pos_idx': [p.reshape(-1) for p in pos_idx.cpu().split(
        list(gt_groups), dim=1)], 'dn_num_group': num_group, 'dn_num_split':
        [num_dn, num_queries]}
    return padding_cls.to(class_embed.device), padding_bbox.to(class_embed.
        device), attn_mask.to(class_embed.device), dn_meta
