def _forward_box(self, features, proposals, targets=None):
    """
        Args:
            features, targets: the same as in
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".
        """
    features = [features[f] for f in self.box_in_features]
    head_outputs = []
    prev_pred_boxes = None
    image_sizes = [x.image_size for x in proposals]
    for k in range(self.num_cascade_stages):
        if k > 0:
            proposals = self._create_proposals_from_boxes(prev_pred_boxes,
                image_sizes)
            if self.training:
                proposals = self._match_and_label_boxes(proposals, k, targets)
        predictions = self._run_stage(features, proposals, k)
        prev_pred_boxes = self.box_predictor[k].predict_boxes(predictions,
            proposals)
        head_outputs.append((self.box_predictor[k], predictions, proposals))
    if self.training:
        losses = {}
        storage = get_event_storage()
        for stage, (predictor, predictions, proposals) in enumerate(
            head_outputs):
            with storage.name_scope('stage{}'.format(stage)):
                stage_losses = predictor.losses(predictions, proposals)
            losses.update({(k + '_stage{}'.format(stage)): v for k, v in
                stage_losses.items()})
        return losses
    else:
        scores_per_stage = [h[0].predict_probs(h[1], h[2]) for h in
            head_outputs]
        scores = [(sum(list(scores_per_image)) * (1.0 / self.
            num_cascade_stages)) for scores_per_image in zip(*scores_per_stage)
            ]
        predictor, predictions, proposals = head_outputs[-1]
        boxes = predictor.predict_boxes(predictions, proposals)
        pred_instances, _ = fast_rcnn_inference(boxes, scores, image_sizes,
            predictor.test_score_thresh, predictor.test_nms_thresh,
            predictor.test_topk_per_image)
        return pred_instances
