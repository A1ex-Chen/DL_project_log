def call(self, fpn_feats, img_info, proposals, gt_bboxes=None, gt_labels=
    None, gt_masks=None, training=True):
    model_outputs = dict()
    if training:
        box_targets, class_targets, rpn_box_rois, proposal_to_label_map = (self
            .bbox_sampler(proposals, gt_bboxes, gt_labels))
    else:
        rpn_box_rois = proposals
    box_roi_features = self.bbox_roi_extractor(fpn_feats, rpn_box_rois)
    class_outputs, box_outputs, _ = self.bbox_head(inputs=box_roi_features)
    if not training:
        model_outputs.update(self.inference_detector(class_outputs,
            box_outputs, rpn_box_rois, img_info))
        model_outputs.update({'class_outputs': tf.nn.softmax(class_outputs),
            'box_outputs': box_outputs, 'anchor_boxes': rpn_box_rois})
    else:
        if self.bbox_head.loss.box_loss_type not in ['giou', 'ciou']:
            encoded_box_targets = self.box_encoder(boxes=rpn_box_rois,
                gt_bboxes=box_targets, gt_labels=class_targets)
        model_outputs.update({'class_outputs': class_outputs, 'box_outputs':
            box_outputs, 'class_targets': class_targets, 'box_targets': 
            encoded_box_targets if self.bbox_head.loss.box_loss_type not in
            ['giou', 'ciou'] else box_targets, 'box_rois': rpn_box_rois})
        total_loss, class_loss, box_loss = self.bbox_head.loss(model_outputs
            ['class_outputs'], model_outputs['box_outputs'], model_outputs[
            'class_targets'], model_outputs['box_targets'], model_outputs[
            'box_rois'], img_info)
        model_outputs.update({'total_loss_bbox': total_loss, 'class_loss':
            class_loss, 'box_loss': box_loss})
    if not self.with_mask:
        return model_outputs
    if not training:
        return self.call_mask(model_outputs, fpn_feats, training=False)
    max_fg = int(self.bbox_sampler.batch_size_per_im * self.bbox_sampler.
        fg_fraction)
    return self.call_mask(model_outputs, fpn_feats, class_targets=
        class_targets, box_targets=box_targets, rpn_box_rois=rpn_box_rois,
        proposal_to_label_map=proposal_to_label_map, gt_masks=gt_masks,
        max_fg=max_fg, training=True)
