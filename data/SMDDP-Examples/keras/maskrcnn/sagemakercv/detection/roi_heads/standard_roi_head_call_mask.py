def call_mask(self, model_outputs, fpn_feats, class_targets=None,
    box_targets=None, rpn_box_rois=None, proposal_to_label_map=None,
    gt_masks=None, max_fg=None, training=True):
    if not training:
        selected_box_rois = model_outputs['detection_boxes']
        class_indices = model_outputs['detection_classes']
        class_indices = tf.cast(class_indices, dtype=tf.int32)
    else:
        (selected_class_targets, selected_box_targets, selected_box_rois,
            proposal_to_label_map) = (training_ops.select_fg_for_masks(
            class_targets=class_targets, box_targets=box_targets, boxes=
            rpn_box_rois, proposal_to_label_map=proposal_to_label_map,
            max_num_fg=max_fg))
        class_indices = selected_class_targets
        class_indices = tf.cast(selected_class_targets, dtype=tf.int32)
    mask_roi_features = self.mask_roi_extractor(fpn_feats, selected_box_rois)
    mask_outputs = self.mask_head(inputs=mask_roi_features, class_indices=
        class_indices)
    if training:
        mask_targets = training_ops.get_mask_targets(fg_boxes=
            selected_box_rois, fg_proposal_to_label_map=
            proposal_to_label_map, fg_box_targets=selected_box_targets,
            mask_gt_labels=gt_masks, output_size=self.mask_head.
            _mrcnn_resolution)
        model_outputs.update({'mask_outputs': mask_outputs, 'mask_targets':
            mask_targets, 'selected_class_targets': selected_class_targets})
        mask_loss = self.mask_head.loss(model_outputs['mask_outputs'],
            model_outputs['mask_targets'], model_outputs[
            'selected_class_targets'])
        model_outputs.update({'mask_loss': mask_loss})
    else:
        model_outputs.update({'detection_masks': tf.nn.sigmoid(mask_outputs)})
    return model_outputs
