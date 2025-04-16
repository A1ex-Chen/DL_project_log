def inference(self, retina_box_cls, retina_box_delta, retina_anchors,
    small_det_logits, small_det_delta, small_det_anchors, image_sizes):
    results = []
    N, _, _, _ = retina_box_cls[0].size()
    retina_box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in
        retina_box_cls]
    retina_box_delta = [permute_to_N_HWA_K(x, 4) for x in retina_box_delta]
    small_det_logits = [x.view(N, -1, self.num_classes) for x in
        small_det_logits]
    small_det_delta = [x.view(N, -1, 4) for x in small_det_delta]
    for img_idx, image_size in enumerate(image_sizes):
        retina_box_cls_per_image = [box_cls_per_level[img_idx] for
            box_cls_per_level in retina_box_cls]
        retina_box_reg_per_image = [box_reg_per_level[img_idx] for
            box_reg_per_level in retina_box_delta]
        small_det_logits_per_image = [small_det_cls_per_level[img_idx] for
            small_det_cls_per_level in small_det_logits]
        small_det_reg_per_image = [small_det_reg_per_level[img_idx] for
            small_det_reg_per_level in small_det_delta]
        if len(small_det_anchors) == 0 or type(small_det_anchors[0]
            ) == torch.Tensor:
            small_det_anchor_per_image = [small_det_anchor_per_level[
                img_idx] for small_det_anchor_per_level in small_det_anchors]
        else:
            small_det_anchor_per_image = small_det_anchors
        results_per_img = self.inference_single_image(retina_box_cls_per_image,
            retina_box_reg_per_image, retina_anchors,
            small_det_logits_per_image, small_det_reg_per_image,
            small_det_anchor_per_image, tuple(image_size))
        results.append(results_per_img)
    return results
