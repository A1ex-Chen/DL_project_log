def _rescale_detected_boxes(self, augmented_inputs, merged_instances, tfms):
    augmented_instances = []
    for input, tfm in zip(augmented_inputs, tfms):
        pred_boxes = merged_instances.pred_boxes.tensor.cpu().numpy()
        pred_boxes = torch.from_numpy(tfm.apply_box(pred_boxes))
        aug_instances = Instances(image_size=input['image'].shape[1:3],
            pred_boxes=Boxes(pred_boxes), pred_classes=merged_instances.
            pred_classes, scores=merged_instances.scores)
        augmented_instances.append(aug_instances)
    return augmented_instances
