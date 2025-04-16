def train_forward(self, batched_inputs, just_forward=False):
    if self.clear_cuda_cache:
        torch.cuda.empty_cache()
    if 'instances' in batched_inputs[0]:
        gt_instances = [x['instances'].to(self.device) for x in batched_inputs]
    elif 'targets' in batched_inputs[0]:
        log_first_n(logging.WARN,
            "'targets' in the model inputs is now renamed to 'instances'!",
            n=10)
        gt_instances = [x['targets'].to(self.device) for x in batched_inputs]
    else:
        gt_instances = None
    images = self.preprocess_image(batched_inputs)
    features = self.backbone(images.tensor)
    all_features = [features[f] for f in self.in_features]
    all_anchors, all_centers = self.anchor_generator(all_features)
    query_feature = [all_features[x] for x in self.query_layer_train]
    _, query_centers = self.query_anchor_generator(query_feature)
    det_cls, det_delta = self.det_head(all_features)
    query_logits = self.query_head(query_feature)
    if just_forward:
        return None
    gt_classes, gt_reg_targets = self.get_det_gt(all_anchors, gt_instances)
    losses = self.det_loss(gt_classes, gt_reg_targets, det_cls, det_delta,
        all_anchors)
    gt_query = self.get_query_gt(query_centers, gt_instances)
    query_forgrounds = [gt.sum().item() for gt in gt_query]
    _query_loss = self.query_loss(gt_query, query_logits, self.
        query_loss_gammas, self.query_loss_weights)
    losses.update(_query_loss)
    return losses
