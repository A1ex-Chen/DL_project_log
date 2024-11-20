def forward(self, example):
    """module's forward should always accept dict and return loss.
        """
    voxels = example['voxels']
    num_points = example['num_points']
    coors = example['coordinates']
    batch_anchors = example['anchors']
    batch_size_dev = batch_anchors.shape[0]
    t = time.time()
    voxel_features = self.voxel_feature_extractor(voxels, num_points)
    spatial_features = self.tdbn_feature_extractor(voxel_features, coors,
        batch_size_dev)
    preds_dict = self.det_net(spatial_features)
    box_preds = preds_dict['box_preds']
    cls_preds = preds_dict['cls_preds']
    self._total_forward_time += time.time() - t
    if self.training:
        labels = example['labels']
        reg_targets = example['reg_targets']
        cls_weights, reg_weights, cared = prepare_loss_weights(labels,
            pos_cls_weight=self._pos_cls_weight, neg_cls_weight=self.
            _neg_cls_weight, loss_norm_type=self._loss_norm_type, dtype=
            voxels.dtype)
        cls_targets = labels * cared.type_as(labels)
        cls_targets = cls_targets.unsqueeze(-1)
        loc_loss, cls_loss = create_loss(self._loc_loss_ftor, self.
            _cls_loss_ftor, box_preds=box_preds, cls_preds=cls_preds,
            cls_targets=cls_targets, cls_weights=cls_weights, reg_targets=
            reg_targets, reg_weights=reg_weights, num_class=self._num_class,
            encode_rad_error_by_sin=self._encode_rad_error_by_sin,
            encode_background_as_zeros=self._encode_background_as_zeros,
            box_code_size=self._box_coder.code_size)
        loc_loss_reduced = loc_loss.sum() / batch_size_dev
        loc_loss_reduced *= self._loc_loss_weight
        cls_pos_loss, cls_neg_loss = _get_pos_neg_loss(cls_loss, labels)
        cls_pos_loss /= self._pos_cls_weight
        cls_neg_loss /= self._neg_cls_weight
        cls_loss_reduced = cls_loss.sum() / batch_size_dev
        cls_loss_reduced *= self._cls_loss_weight
        loss = loc_loss_reduced + cls_loss_reduced
        if self._use_direction_classifier:
            dir_targets = get_direction_target(example['anchors'], reg_targets)
            dir_logits = preds_dict['dir_cls_preds'].view(batch_size_dev, -1, 2
                )
            weights = (labels > 0).type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self._dir_loss_ftor(dir_logits, dir_targets, weights
                =weights)
            dir_loss = dir_loss.sum() / batch_size_dev
            loss += dir_loss * self._direction_loss_weight
        return {'loss': loss, 'cls_loss': cls_loss, 'loc_loss': loc_loss,
            'cls_pos_loss': cls_pos_loss, 'cls_neg_loss': cls_neg_loss,
            'cls_preds': cls_preds, 'dir_loss_reduced': dir_loss,
            'cls_loss_reduced': cls_loss_reduced, 'loc_loss_reduced':
            loc_loss_reduced, 'cared': cared}
    else:
        return self.predict(example, preds_dict)
