def pred_to_json(self, predn, filename):
    """Serialize YOLO predictions to COCO json format."""
    stem = Path(filename).stem
    image_id = int(stem) if stem.isnumeric() else stem
    rbox = torch.cat([predn[:, :4], predn[:, -1:]], dim=-1)
    poly = ops.xywhr2xyxyxyxy(rbox).view(-1, 8)
    for i, (r, b) in enumerate(zip(rbox.tolist(), poly.tolist())):
        self.jdict.append({'image_id': image_id, 'category_id': self.
            class_map[int(predn[i, 5].item())], 'score': round(predn[i, 4].
            item(), 5), 'rbox': [round(x, 3) for x in r], 'poly': [round(x,
            3) for x in b]})
