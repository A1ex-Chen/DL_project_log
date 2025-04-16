def save_one_json(predn, jdict, path, class_map):
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])
    box[:, :2] -= box[:, 2:] / 2
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({'image_id': image_id, 'category_id': class_map[int(p[
            5])], 'bbox': [round(x, 3) for x in b], 'score': round(p[4], 5)})
