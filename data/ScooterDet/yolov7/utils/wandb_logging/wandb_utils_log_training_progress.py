def log_training_progress(self, predn, path, names):
    if self.val_table and self.result_table:
        class_set = wandb.Classes([{'id': id, 'name': name} for id, name in
            names.items()])
        box_data = []
        total_conf = 0
        for *xyxy, conf, cls in predn.tolist():
            if conf >= 0.25:
                box_data.append({'position': {'minX': xyxy[0], 'minY': xyxy
                    [1], 'maxX': xyxy[2], 'maxY': xyxy[3]}, 'class_id': int
                    (cls), 'box_caption': '%s %.3f' % (names[cls], conf),
                    'scores': {'class_score': conf}, 'domain': 'pixel'})
                total_conf = total_conf + conf
        boxes = {'predictions': {'box_data': box_data, 'class_labels': names}}
        id = self.val_table_map[Path(path).name]
        self.result_table.add_data(self.current_epoch, id, wandb.Image(self
            .val_table.data[id][1], boxes=boxes, classes=class_set), 
            total_conf / max(1, len(box_data)))
