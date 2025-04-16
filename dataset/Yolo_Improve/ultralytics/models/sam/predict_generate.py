def generate(self, im, crop_n_layers=0, crop_overlap_ratio=512 / 1500,
    crop_downscale_factor=1, point_grids=None, points_stride=32,
    points_batch_size=64, conf_thres=0.88, stability_score_thresh=0.95,
    stability_score_offset=0.95, crop_nms_thresh=0.7):
    """
        Perform image segmentation using the Segment Anything Model (SAM).

        This function segments an entire image into constituent parts by leveraging SAM's advanced architecture
        and real-time performance capabilities. It can optionally work on image crops for finer segmentation.

        Args:
            im (torch.Tensor): Input tensor representing the preprocessed image with dimensions (N, C, H, W).
            crop_n_layers (int): Specifies the number of layers for additional mask predictions on image crops.
                                 Each layer produces 2**i_layer number of image crops.
            crop_overlap_ratio (float): Determines the overlap between crops. Scaled down in subsequent layers.
            crop_downscale_factor (int): Scaling factor for the number of sampled points-per-side in each layer.
            point_grids (list[np.ndarray], optional): Custom grids for point sampling normalized to [0,1].
                                                      Used in the nth crop layer.
            points_stride (int, optional): Number of points to sample along each side of the image.
                                           Exclusive with 'point_grids'.
            points_batch_size (int): Batch size for the number of points processed simultaneously.
            conf_thres (float): Confidence threshold [0,1] for filtering based on the model's mask quality prediction.
            stability_score_thresh (float): Stability threshold [0,1] for mask filtering based on mask stability.
            stability_score_offset (float): Offset value for calculating stability score.
            crop_nms_thresh (float): IoU cutoff for NMS to remove duplicate masks between crops.

        Returns:
            (tuple): A tuple containing segmented masks, confidence scores, and bounding boxes.
        """
    import torchvision
    self.segment_all = True
    ih, iw = im.shape[2:]
    crop_regions, layer_idxs = generate_crop_boxes((ih, iw), crop_n_layers,
        crop_overlap_ratio)
    if point_grids is None:
        point_grids = build_all_layer_point_grids(points_stride,
            crop_n_layers, crop_downscale_factor)
    pred_masks, pred_scores, pred_bboxes, region_areas = [], [], [], []
    for crop_region, layer_idx in zip(crop_regions, layer_idxs):
        x1, y1, x2, y2 = crop_region
        w, h = x2 - x1, y2 - y1
        area = torch.tensor(w * h, device=im.device)
        points_scale = np.array([[w, h]])
        crop_im = F.interpolate(im[..., y1:y2, x1:x2], (ih, iw), mode=
            'bilinear', align_corners=False)
        points_for_image = point_grids[layer_idx] * points_scale
        crop_masks, crop_scores, crop_bboxes = [], [], []
        for points, in batch_iterator(points_batch_size, points_for_image):
            pred_mask, pred_score = self.prompt_inference(crop_im, points=
                points, multimask_output=True)
            pred_mask = F.interpolate(pred_mask[None], (h, w), mode=
                'bilinear', align_corners=False)[0]
            idx = pred_score > conf_thres
            pred_mask, pred_score = pred_mask[idx], pred_score[idx]
            stability_score = calculate_stability_score(pred_mask, self.
                model.mask_threshold, stability_score_offset)
            idx = stability_score > stability_score_thresh
            pred_mask, pred_score = pred_mask[idx], pred_score[idx]
            pred_mask = pred_mask > self.model.mask_threshold
            pred_bbox = batched_mask_to_box(pred_mask).float()
            keep_mask = ~is_box_near_crop_edge(pred_bbox, crop_region, [0, 
                0, iw, ih])
            if not torch.all(keep_mask):
                pred_bbox, pred_mask, pred_score = pred_bbox[keep_mask
                    ], pred_mask[keep_mask], pred_score[keep_mask]
            crop_masks.append(pred_mask)
            crop_bboxes.append(pred_bbox)
            crop_scores.append(pred_score)
        crop_masks = torch.cat(crop_masks)
        crop_bboxes = torch.cat(crop_bboxes)
        crop_scores = torch.cat(crop_scores)
        keep = torchvision.ops.nms(crop_bboxes, crop_scores, self.args.iou)
        crop_bboxes = uncrop_boxes_xyxy(crop_bboxes[keep], crop_region)
        crop_masks = uncrop_masks(crop_masks[keep], crop_region, ih, iw)
        crop_scores = crop_scores[keep]
        pred_masks.append(crop_masks)
        pred_bboxes.append(crop_bboxes)
        pred_scores.append(crop_scores)
        region_areas.append(area.expand(len(crop_masks)))
    pred_masks = torch.cat(pred_masks)
    pred_bboxes = torch.cat(pred_bboxes)
    pred_scores = torch.cat(pred_scores)
    region_areas = torch.cat(region_areas)
    if len(crop_regions) > 1:
        scores = 1 / region_areas
        keep = torchvision.ops.nms(pred_bboxes, scores, crop_nms_thresh)
        pred_masks, pred_bboxes, pred_scores = pred_masks[keep], pred_bboxes[
            keep], pred_scores[keep]
    return pred_masks, pred_scores, pred_bboxes
