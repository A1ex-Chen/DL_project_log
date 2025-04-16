def log_training_progress(self, predn, path, names):
    """
        Build evaluation Table. Uses reference from validation dataset table.

        arguments:
        predn (list): list of predictions in the native space in the format - [xmin, ymin, xmax, ymax, confidence, class]
        path (str): local path of the current evaluation image
        names (dict(int, str)): hash map that maps class ids to labels
        """
    class_set = wandb.Classes([{'id': id, 'name': name} for id, name in
        names.items()])
    box_data = []
    avg_conf_per_class = [0] * len(self.data_dict['names'])
    pred_class_count = {}
    for *xyxy, conf, cls in predn.tolist():
        if conf >= 0.25:
            cls = int(cls)
            box_data.append({'position': {'minX': xyxy[0], 'minY': xyxy[1],
                'maxX': xyxy[2], 'maxY': xyxy[3]}, 'class_id': cls,
                'box_caption': f'{names[cls]} {conf:.3f}', 'scores': {
                'class_score': conf}, 'domain': 'pixel'})
            avg_conf_per_class[cls] += conf
            if cls in pred_class_count:
                pred_class_count[cls] += 1
            else:
                pred_class_count[cls] = 1
    for pred_class in pred_class_count.keys():
        avg_conf_per_class[pred_class] = avg_conf_per_class[pred_class
            ] / pred_class_count[pred_class]
    boxes = {'predictions': {'box_data': box_data, 'class_labels': names}}
    id = self.val_table_path_map[Path(path).name]
    self.result_table.add_data(self.current_epoch, id, self.val_table.data[
        id][1], wandb.Image(self.val_table.data[id][1], boxes=boxes,
        classes=class_set), *avg_conf_per_class)
