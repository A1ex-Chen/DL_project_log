def call(self, features, labels=None, training=True, weight_decay=0.0):
    x = self.backbone(features['images'], training=training)
    feature_maps = self.neck(x, training=training)
    scores_outputs, box_outputs, proposals = self.rpn_head(feature_maps,
        features['image_info'], training=training)
    model_outputs = {'images': features['images'], 'image_info': features[
        'image_info']}
    if training:
        model_outputs.update(self.roi_head(feature_maps, features[
            'image_info'], proposals[0], gt_bboxes=labels['gt_boxes'],
            gt_labels=labels['gt_classes'], gt_masks=labels.get(
            'cropped_gt_masks', None), training=training))
        total_rpn_loss, rpn_score_loss, rpn_box_loss = self.rpn_head.loss(
            scores_outputs, box_outputs, labels)
        model_outputs.update({'total_rpn_loss': total_rpn_loss,
            'rpn_score_loss': rpn_score_loss, 'rpn_box_loss': rpn_box_loss})
        loss_dict = self.parse_losses(model_outputs, weight_decay=weight_decay)
        model_outputs['total_loss'] = loss_dict['total_loss']
        if weight_decay > 0.0:
            model_outputs['l2_loss'] = loss_dict['l2_loss']
    else:
        model_outputs.update(self.roi_head(feature_maps, features[
            'image_info'], proposals[0], training=training))
    return model_outputs
