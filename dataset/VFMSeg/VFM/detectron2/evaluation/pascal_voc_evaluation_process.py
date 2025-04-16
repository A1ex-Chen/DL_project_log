def process(self, inputs, outputs):
    for input, output in zip(inputs, outputs):
        image_id = input['image_id']
        instances = output['instances'].to(self._cpu_device)
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.tolist()
        classes = instances.pred_classes.tolist()
        for box, score, cls in zip(boxes, scores, classes):
            xmin, ymin, xmax, ymax = box
            xmin += 1
            ymin += 1
            self._predictions[cls].append(
                f'{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}'
                )
