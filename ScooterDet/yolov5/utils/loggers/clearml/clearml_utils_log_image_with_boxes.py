def log_image_with_boxes(self, image_path, boxes, class_names, image,
    conf_threshold=0.25):
    """
        Draw the bounding boxes on a single image and report the result as a ClearML debug sample.

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
            im = np.ascontiguousarray(np.moveaxis(image.mul(255).clamp(0, 
                255).byte().cpu().numpy(), 0, 2))
            annotator = Annotator(im=im, pil=True)
            for i, (conf, class_nr, box) in enumerate(zip(boxes[:, 4],
                boxes[:, 5], boxes[:, :4])):
                color = colors(i)
                class_name = class_names[int(class_nr)]
                confidence_percentage = round(float(conf) * 100, 2)
                label = f'{class_name}: {confidence_percentage}%'
                if conf > conf_threshold:
                    annotator.rectangle(box.cpu().numpy(), outline=color)
                    annotator.box_label(box.cpu().numpy(), label=label,
                        color=color)
            annotated_image = annotator.result()
            self.task.get_logger().report_image(title='Bounding Boxes',
                series=image_path.name, iteration=self.current_epoch, image
                =annotated_image)
            self.current_epoch_logged_images.add(image_path)
