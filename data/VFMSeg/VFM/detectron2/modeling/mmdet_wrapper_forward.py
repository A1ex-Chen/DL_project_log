def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
    images = [x['image'].to(self.device) for x in batched_inputs]
    images = [((x - self.pixel_mean) / self.pixel_std) for x in images]
    images = ImageList.from_tensors(images, size_divisibility=self.
        size_divisibility).tensor
    metas = []
    rescale = {('height' in x) for x in batched_inputs}
    if len(rescale) != 1:
        raise ValueError(
            "Some inputs have original height/width, but some don't!")
    rescale = list(rescale)[0]
    output_shapes = []
    for input in batched_inputs:
        meta = {}
        c, h, w = input['image'].shape
        meta['img_shape'] = meta['ori_shape'] = h, w, c
        if rescale:
            scale_factor = np.array([w / input['width'], h / input['height'
                ]] * 2, dtype='float32')
            ori_shape = input['height'], input['width']
            output_shapes.append(ori_shape)
            meta['ori_shape'] = ori_shape + (c,)
        else:
            scale_factor = 1.0
            output_shapes.append((h, w))
        meta['scale_factor'] = scale_factor
        meta['flip'] = False
        padh, padw = images.shape[-2:]
        meta['pad_shape'] = padh, padw, c
        metas.append(meta)
    if self.training:
        gt_instances = [x['instances'].to(self.device) for x in batched_inputs]
        if gt_instances[0].has('gt_masks'):
            from mmdet.core import PolygonMasks as mm_PolygonMasks, BitmapMasks as mm_BitMasks

            def convert_mask(m, shape):
                if isinstance(m, BitMasks):
                    return mm_BitMasks(m.tensor.cpu().numpy(), shape[0],
                        shape[1])
                else:
                    return mm_PolygonMasks(m.polygons, shape[0], shape[1])
            gt_masks = [convert_mask(x.gt_masks, x.image_size) for x in
                gt_instances]
            losses_and_metrics = self.detector.forward_train(images, metas,
                [x.gt_boxes.tensor for x in gt_instances], [x.gt_classes for
                x in gt_instances], gt_masks=gt_masks)
        else:
            losses_and_metrics = self.detector.forward_train(images, metas,
                [x.gt_boxes.tensor for x in gt_instances], [x.gt_classes for
                x in gt_instances])
        return _parse_losses(losses_and_metrics)
    else:
        results = self.detector.simple_test(images, metas, rescale=rescale)
        results = [{'instances': _convert_mmdet_result(r, shape)} for r,
            shape in zip(results, output_shapes)]
        return results
