def _get_augmented_boxes(self, augmented_inputs, tfms):
    outputs = self._batch_inference(augmented_inputs)
    all_boxes = []
    all_scores = []
    all_classes = []
    for output, tfm in zip(outputs, tfms):
        pred_boxes = output.pred_boxes.tensor
        original_pred_boxes = tfm.inverse().apply_box(pred_boxes.cpu().numpy())
        all_boxes.append(torch.from_numpy(original_pred_boxes).to(
            pred_boxes.device))
        all_scores.extend(output.scores)
        all_classes.extend(output.pred_classes)
    all_boxes = torch.cat(all_boxes, dim=0)
    return all_boxes, all_scores, all_classes
