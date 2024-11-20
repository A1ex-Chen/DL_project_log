def log_image_with_boxes(self, image_path, boxes, class_names, image):
    """
        Draw the bounding boxes on a single image and report the result as a ClearML debug sample

        arguments:
        image_path (PosixPath) the path the original image file
        boxes (list): list of scaled predictions in the format - [xmin, ymin, xmax, ymax, confidence, class]
        class_names (dict): dict containing mapping of class int to class name
        image (Tensor): A torch tensor containing the actual image data
        """
    if len(self.current_epoch_logged_images
        ) < self.max_imgs_to_log_per_epoch and self.current_epoch >= 0:
        if (self.current_epoch % self.bbox_interval == 0 and image_path not in
            self.current_epoch_logged_images):
            converter = ToPILImage()
            labels = []
            for conf, class_nr in zip(boxes[:, 4], boxes[:, 5]):
                class_name = class_names[int(class_nr)]
                confidence = round(float(conf) * 100, 2)
                labels.append(f'{class_name}: {confidence}%')
            annotated_image = converter(draw_bounding_boxes(image=image.mul
                (255).clamp(0, 255).byte().cpu(), boxes=boxes[:, :4],
                labels=labels))
            self.task.get_logger().report_image(title='Bounding Boxes',
                series=image_path.name, iteration=self.current_epoch, image
                =annotated_image)
            self.current_epoch_logged_images.add(image_path)
