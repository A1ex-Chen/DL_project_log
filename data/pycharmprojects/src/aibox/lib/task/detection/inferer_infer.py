@torch.no_grad()
def infer(self, image_batch: List[Tensor], lower_prob_thresh: float,
    upper_prob_thresh: float) ->Inference:
    image_batch = Bunch(image_batch)
    (anchor_bboxes_batch, proposal_bboxes_batch, proposal_probs_batch,
        detection_bboxes_batch, detection_classes_batch, detection_probs_batch
        ) = self._model.eval().forward(image_batch)
    final_detection_bboxes_batch = []
    final_detection_classes_batch = []
    final_detection_probs_batch = []
    for detection_bboxes, detection_classes, detection_probs in zip(
        detection_bboxes_batch, detection_classes_batch, detection_probs_batch
        ):
        kept_mask = (detection_probs >= lower_prob_thresh) & (detection_probs
             <= upper_prob_thresh)
        final_detection_bboxes = detection_bboxes[kept_mask]
        final_detection_classes = detection_classes[kept_mask]
        final_detection_probs = detection_probs[kept_mask]
        kept_indices = remove_small_boxes(final_detection_bboxes, 1)
        final_detection_bboxes = final_detection_bboxes[kept_indices]
        final_detection_classes = final_detection_classes[kept_indices]
        final_detection_probs = final_detection_probs[kept_indices]
        final_detection_bboxes_batch.append(final_detection_bboxes)
        final_detection_classes_batch.append(final_detection_classes)
        final_detection_probs_batch.append(final_detection_probs)
    inference = Inferer.Inference(anchor_bboxes_batch,
        proposal_bboxes_batch, proposal_probs_batch, detection_bboxes_batch,
        detection_classes_batch, detection_probs_batch,
        final_detection_bboxes_batch, final_detection_classes_batch,
        final_detection_probs_batch)
    return inference
