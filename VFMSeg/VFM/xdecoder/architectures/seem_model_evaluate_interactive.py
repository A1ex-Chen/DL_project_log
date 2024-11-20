def evaluate_interactive(self, batched_inputs):
    assert self.task_switch['spatial']
    assert 'spatial_query' in batched_inputs[0]
    assert len(batched_inputs) == 1, 'only support batch size equal to 1'
    images = [x['image'].to(self.device) for x in batched_inputs]
    images = [((x - self.pixel_mean) / self.pixel_std) for x in images]
    images = ImageList.from_tensors(images, self.size_divisibility)
    img_bs = images.tensor.shape[0]
    targets = targets_grounding = queries_grounding = None
    extra = {}
    features = self.backbone(images.tensor)
    mask_features, transformer_encoder_features, multi_scale_features = (self
        .sem_seg_head.pixel_decoder.forward_features(features))
    image_sizes = [x['image'].shape[-2:] for x in batched_inputs]
    nm = len(batched_inputs[0]['spatial_query']['rand_shape'])
    multi_scale_features = [m.repeat(nm, 1, 1, 1) for m in multi_scale_features
        ]
    mask_features = mask_features.repeat(nm, 1, 1, 1)
    all_batch_shape_iou = []
    pred_smask_pointer = None
    prev_smask_pointer = None
    pred_smask_all = None
    query_index = self.sem_seg_head.predictor.query_index
    assert self.interactive_mode == 'best'
    pos_masks = batched_inputs[0]['spatial_query']['rand_shape'].to(self.device
        ).unbind(0)
    pos_masks = ImageList.from_tensors(pos_masks, self.size_divisibility
        ).tensor.unbind(0)
    neg_masks = (batched_inputs[0]['spatial_query']['rand_shape'].to(self.
        device) & False).unbind(0)
    neg_masks = ImageList.from_tensors(neg_masks, self.size_divisibility
        ).tensor.unbind(0)
    extra.update({'spatial_query_pos_mask': pos_masks,
        'spatial_query_neg_mask': neg_masks})
    for i in range(self.interactive_iter):
        outputs = self.sem_seg_head.predictor(multi_scale_features,
            mask_features, target_queries=queries_grounding, extra=extra,
            task='spatial')
        extra.update(outputs)
        pred_smask = F.interpolate(outputs['prev_mask'], images.tensor.
            shape[-2:], mode='bicubic')
        s = image_sizes[0]
        b = batched_inputs[0]
        pred_smask_all = F.interpolate(pred_smask[:, :, :s[0], :s[1]], (b[
            'height'], b['width']), mode='bicubic')[:, 0].sigmoid() > 0.5
        gt_smask = b['gt_masks_orisize']
        all_batch_shape_iou += [get_iou(gt_smask, pred_smask_all)]
        extra.update(self.prepare_next_spaital_mask(extra, batched_inputs))
    all_batch_shape_iou = torch.stack(all_batch_shape_iou)
    processed_results = [{'mask_iou': all_batch_shape_iou[:, i]} for i in
        range(len(all_batch_shape_iou[0]))]
    return processed_results
