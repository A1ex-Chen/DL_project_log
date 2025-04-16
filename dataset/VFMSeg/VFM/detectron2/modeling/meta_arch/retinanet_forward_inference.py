def forward_inference(self, images: ImageList, features: List[Tensor],
    predictions: List[List[Tensor]]):
    pred_logits, pred_anchor_deltas = self._transpose_dense_predictions(
        predictions, [self.num_classes, 4])
    anchors = self.anchor_generator(features)
    results: List[Instances] = []
    for img_idx, image_size in enumerate(images.image_sizes):
        scores_per_image = [x[img_idx].sigmoid_() for x in pred_logits]
        deltas_per_image = [x[img_idx] for x in pred_anchor_deltas]
        results_per_image = self.inference_single_image(anchors,
            scores_per_image, deltas_per_image, image_size)
        results.append(results_per_image)
    return results
