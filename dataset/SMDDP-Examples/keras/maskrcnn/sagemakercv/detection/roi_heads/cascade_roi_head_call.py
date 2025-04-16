def call(self, fpn_feats, img_info, proposals, gt_bboxes=None, gt_labels=
    None, gt_masks=None, training=True):
    model_outputs = dict()
    for stage in range(self.num_stages):
        if training:
            (box_targets, class_targets, rpn_box_rois, proposal_to_label_map
                ) = self.bbox_sampler[stage](proposals, gt_bboxes, gt_labels)
        else:
            rpn_box_rois = proposals
        box_roi_features = self.bbox_roi_extractor(fpn_feats, rpn_box_rois)
        if training:
            box_roi_features = self.scale_gradient(box_roi_features)
        class_outputs, box_outputs, _ = self.bbox_head[stage](inputs=
            box_roi_features)
        if training:
            if self.bbox_head[stage].loss.box_loss_type not in ['giou', 'ciou'
                ]:
                encoded_box_targets = self.box_encoder[stage](boxes=
                    rpn_box_rois, gt_bboxes=box_targets, gt_labels=
                    class_targets)
            model_outputs.update({f'class_outputs_{stage}': class_outputs,
                f'box_outputs_{stage}': box_outputs,
                f'class_targets_{stage}': class_targets,
                f'box_targets_{stage}': encoded_box_targets,
                f'box_rois_{stage}': rpn_box_rois})
            total_loss, class_loss, box_loss = self.bbox_head[stage].loss(
                model_outputs[f'class_outputs_{stage}'], model_outputs[
                f'box_outputs_{stage}'], model_outputs[
                f'class_targets_{stage}'], model_outputs[
                f'box_targets_{stage}'], model_outputs[f'box_rois_{stage}'],
                img_info)
            model_outputs.update({f'total_loss_{stage}': total_loss,
                f'class_loss_{stage}': class_loss, f'box_loss_{stage}':
                box_loss})
        if stage < self.num_stages - 1:
            new_proposals = box_utils.decode_boxes(box_outputs[:, :, 4:],
                rpn_box_rois, weights=self.bbox_head[stage].loss.
                bbox_reg_weights)
            height = tf.expand_dims(img_info[:, 0:1], axis=-1)
            width = tf.expand_dims(img_info[:, 1:2], axis=-1)
            boxes = box_utils.clip_boxes(new_proposals, (height, width))
            proposals = tf.stop_gradient(new_proposals)
    if not training:
        model_outputs.update(self.inference_detector(class_outputs,
            box_outputs, rpn_box_rois, img_info))
        model_outputs.update({'class_outputs': tf.nn.softmax(class_outputs),
            'box_outputs': box_outputs, 'anchor_boxes': rpn_box_rois})
    else:
        model_outputs.update({'class_outputs': class_outputs, 'box_outputs':
            box_outputs, 'class_targets': class_targets, 'box_targets':
            encoded_box_targets, 'box_rois': rpn_box_rois})
    if not self.with_mask:
        return model_outputs
    if not training:
        return self.call_mask(model_outputs, fpn_feats, training=False)
    max_fg = int(self.bbox_sampler[-1].batch_size_per_im * self.
        bbox_sampler[-1].fg_fraction)
    print(class_targets)
    return self.call_mask(model_outputs, fpn_feats, class_targets=
        class_targets, box_targets=box_targets, rpn_box_rois=rpn_box_rois,
        proposal_to_label_map=proposal_to_label_map, gt_masks=gt_masks,
        max_fg=max_fg, training=True)
