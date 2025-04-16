def build_image_batch(self, runner):
    data_batch = runner.data_batch
    outputs = runner.outputs
    detections = self.inference_detector(outputs['class_outputs'], outputs[
        'box_outputs'], outputs['box_rois'], outputs['image_info'])
    boxes = detections['detection_boxes']
    classes = detections['detection_classes']
    scores = detections['detection_scores']
    images = data_batch[0]['images']
    image_info = data_batch[0]['image_info']
    batch_size = boxes.shape[0]
    detection_images = [self.build_image(images[i], image_info[i], boxes[i],
        scores[i], classes[i]) for i in range(batch_size)]
    return detection_images
