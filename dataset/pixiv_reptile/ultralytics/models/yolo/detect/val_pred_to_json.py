def pred_to_json(self, predn, filename):
    """Serialize YOLO predictions to COCO json format."""
    stem = Path(filename).stem
    image_id = int(stem) if stem.isnumeric() else stem
    box = ops.xyxy2xywh(predn[:, :4])
    box[:, :2] -= box[:, 2:] / 2
    for p, b in zip(predn.tolist(), box.tolist()):
        self.jdict.append({'image_id': image_id, 'category_id': self.
            class_map[int(p[5])] + (1 if self.is_lvis else 0), 'bbox': [
            round(x, 3) for x in b], 'score': round(p[4], 5)})
