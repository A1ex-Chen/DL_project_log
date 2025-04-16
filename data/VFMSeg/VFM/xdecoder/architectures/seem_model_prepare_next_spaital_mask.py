def prepare_next_spaital_mask(self, outputs, batched_inputs):
    gt_masks = [batched_inputs[i]['spatial_query']['gt_masks'] for i in
        range(len(batched_inputs))]
    if self.training:
        gt_masks = ImageList.from_tensors(gt_masks, self.size_divisibility
            ).tensor
    else:
        gt_masks = ImageList.from_tensors(gt_masks, self.size_divisibility
            ).tensor.transpose(0, 1)
    pred_masks = F.interpolate(outputs['prev_mask'], size=gt_masks.shape[-2
        :], mode='bilinear', align_corners=False).sigmoid() > 0.5
    prev_masks = torch.stack(outputs['spatial_query_pos_mask']) | torch.stack(
        outputs['spatial_query_neg_mask'])
    fn = gt_masks & ~(gt_masks & pred_masks) & ~prev_masks
    fp = ~gt_masks & pred_masks & ~prev_masks
    iou = (gt_masks & pred_masks).sum(list(range(1, len(fn.shape)))) / ((
        gt_masks | pred_masks).sum(dim=list(range(1, len(fn.shape)))) + 1e-08)
    fn_sum = fn.sum(dim=list(range(1, len(fn.shape))))
    fp_sum = fp.sum(dim=list(range(1, len(fp.shape))))
    is_postive = fn_sum > fp_sum
    select_mask = torch.stack([(fn[i] if is_postive[i] else fp[i]) for i in
        range(len(fn))])
    n, _, h, w = select_mask.shape
    mask_dt = distance_transform((~F.pad(select_mask, pad=(1, 1, 1, 1),
        mode='constant', value=0)).float())[:, :, 1:-1, 1:-1].reshape(n, -1)
    max_xy_idx = torch.stack([torch.arange(n), mask_dt.max(dim=-1)[1].cpu()]
        ).tolist()
    next_mask = torch.zeros(gt_masks.shape, device=torch.cuda.current_device()
        ).bool()
    next_mask = next_mask.view(n, -1)
    next_mask[max_xy_idx] = True
    next_mask = next_mask.reshape((n, 1, h, w)).float()
    dilation = 3
    next_mask = F.conv2d(next_mask, self.dilation_kernel, padding=dilation // 2
        ) > 0
    keep = iou < 0.925
    next_mask = next_mask & keep.view(-1, 1, 1, 1)
    pos_mask = []
    neg_mask = []
    for idx, ip in enumerate(is_postive):
        if ip:
            pos_mask += [outputs['spatial_query_pos_mask'][idx] | next_mask
                [idx]]
            neg_mask += [outputs['spatial_query_neg_mask'][idx]]
        else:
            pos_mask += [outputs['spatial_query_pos_mask'][idx]]
            neg_mask += [outputs['spatial_query_neg_mask'][idx] | next_mask
                [idx]]
    if 'false_positive_mask' in outputs:
        fp = outputs['false_positive_mask'] | fp
    return {'spatial_query_pos_mask': pos_mask, 'spatial_query_neg_mask':
        neg_mask, 'false_positive_mask': fp}
