def log_predictions(self, image, labelsn, path, shape, predn):
    if self.logged_images_count >= self.max_images:
        return
    detections = predn[predn[:, 4] > self.conf_thres]
    iou = box_iou(labelsn[:, 1:], detections[:, :4])
    mask, _ = torch.where(iou > self.iou_thres)
    if len(mask) == 0:
        return
    filtered_detections = detections[mask]
    filtered_labels = labelsn[mask]
    image_id = path.split('/')[-1].split('.')[0]
    image_name = f'{image_id}_curr_epoch_{self.experiment.curr_epoch}'
    if image_name not in self.logged_image_names:
        native_scale_image = PIL.Image.open(path)
        self.log_image(native_scale_image, name=image_name)
        self.logged_image_names.append(image_name)
    metadata = []
    for cls, *xyxy in filtered_labels.tolist():
        metadata.append({'label': f'{self.class_names[int(cls)]}-gt',
            'score': 100, 'box': {'x': xyxy[0], 'y': xyxy[1], 'x2': xyxy[2],
            'y2': xyxy[3]}})
    for *xyxy, conf, cls in filtered_detections.tolist():
        metadata.append({'label': f'{self.class_names[int(cls)]}', 'score':
            conf * 100, 'box': {'x': xyxy[0], 'y': xyxy[1], 'x2': xyxy[2],
            'y2': xyxy[3]}})
    self.metadata_dict[image_name] = metadata
    self.logged_images_count += 1
    return
