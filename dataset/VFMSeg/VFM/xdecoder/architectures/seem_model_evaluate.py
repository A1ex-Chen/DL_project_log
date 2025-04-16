def evaluate(self, batched_inputs):
    images = [x['image'].to(self.device) for x in batched_inputs]
    images = [((x - self.pixel_mean) / self.pixel_std) for x in images]
    images = ImageList.from_tensors(images, self.size_divisibility)
    img_bs = images.tensor.shape[0]
    targets = targets_grounding = queries_grounding = None
    features = self.backbone(images.tensor)
    outputs = self.sem_seg_head(features, target_queries=queries_grounding)
    mask_cls_results = outputs['pred_logits']
    mask_pred_results = outputs['pred_masks']
    box_pred_results = outputs['pred_boxes'] if self.task_switch['bbox'] else [
        None for i in range(len(mask_pred_results))]
    mask_pred_results = F.interpolate(mask_pred_results, size=(images.
        tensor.shape[-2], images.tensor.shape[-1]), mode='bilinear',
        align_corners=False)
    input_size = mask_pred_results.shape[-2:]
    del outputs
    processed_results = []
    for mask_cls_result, mask_pred_result, box_pred_result, input_per_image, image_size in zip(
        mask_cls_results, mask_pred_results, box_pred_results,
        batched_inputs, images.image_sizes):
        height = input_per_image.get('height', image_size[0])
        width = input_per_image.get('width', image_size[1])
        processed_results.append({})
        if self.sem_seg_postprocess_before_inference:
            mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                mask_pred_result, image_size, height, width)
            mask_cls_result = mask_cls_result.to(mask_pred_result)
        if self.panoptic_on:
            panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(
                mask_cls_result, mask_pred_result)
            processed_results[-1]['panoptic_seg'] = panoptic_r
    return processed_results
