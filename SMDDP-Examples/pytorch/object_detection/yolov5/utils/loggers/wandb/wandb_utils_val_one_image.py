def val_one_image(self, pred, predn, path, names, im):
    """
        Log validation data for one image. updates the result Table if validation dataset is uploaded and log bbox media panel

        arguments:
        pred (list): list of scaled predictions in the format - [xmin, ymin, xmax, ymax, confidence, class]
        predn (list): list of predictions in the native space - [xmin, ymin, xmax, ymax, confidence, class]
        path (str): local path of the current evaluation image
        """
    if self.val_table and self.result_table:
        self.log_training_progress(predn, path, names)
    if len(self.bbox_media_panel_images
        ) < self.max_imgs_to_log and self.current_epoch > 0:
        if self.current_epoch % self.bbox_interval == 0:
            box_data = [{'position': {'minX': xyxy[0], 'minY': xyxy[1],
                'maxX': xyxy[2], 'maxY': xyxy[3]}, 'class_id': int(cls),
                'box_caption': f'{names[int(cls)]} {conf:.3f}', 'scores': {
                'class_score': conf}, 'domain': 'pixel'} for *xyxy, conf,
                cls in pred.tolist()]
            boxes = {'predictions': {'box_data': box_data, 'class_labels':
                names}}
            self.bbox_media_panel_images.append(wandb.Image(im, boxes=boxes,
                caption=path.name))
