def evaluate_referring_image(self, batched_inputs, extra={}):
    assert self.task_switch['spatial']
    assert len(batched_inputs) == 1, 'only support batch size equal to 1'
    assert self.interactive_mode == 'best'
    images = [x['image'].to(self.device) for x in batched_inputs]
    images = [((x - self.pixel_mean) / self.pixel_std) for x in images]
    images = ImageList.from_tensors(images, self.size_divisibility)
    img_bs = images.tensor.shape[0]
    targets = targets_grounding = queries_grounding = None
    features = self.backbone(images.tensor)
    mask_features, transformer_encoder_features, multi_scale_features = (self
        .sem_seg_head.pixel_decoder.forward_features(features))
    if 'spatial_query' in batched_inputs[0]:
        image_sizes = [x['image'].shape[-2:] for x in batched_inputs]
        nm = len(batched_inputs[0]['spatial_query']['rand_shape'])
        multi_scale_features = [m.repeat(nm, 1, 1, 1) for m in
            multi_scale_features]
        mask_features = mask_features.repeat(nm, 1, 1, 1)
        query_index = self.sem_seg_head.predictor.query_index
        pos_masks = batched_inputs[0]['spatial_query']['rand_shape'].to(self
            .device).unbind(0)
        pos_masks = ImageList.from_tensors(pos_masks, self.size_divisibility
            ).tensor.unbind(0)
        neg_masks = (batched_inputs[0]['spatial_query']['rand_shape'].to(
            self.device) & False).unbind(0)
        neg_masks = ImageList.from_tensors(neg_masks, self.size_divisibility
            ).tensor.unbind(0)
        extra.update({'spatial_query_pos_mask': pos_masks,
            'spatial_query_neg_mask': neg_masks})
    outputs = self.sem_seg_head.predictor(multi_scale_features,
        mask_features, target_queries=queries_grounding, extra=extra, task=
        'refimg')
    return outputs, images.tensor.shape
