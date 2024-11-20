def build_image(self, image, image_info, boxes, scores, classes):
    image = visualization.restore_image(image, image_info)
    detection_image = visualization.build_image(image, boxes, scores,
        classes, class_names=coco_categories, threshold=self.threshold)
    detection_image = tf.expand_dims(detection_image, axis=0)
    return detection_image
