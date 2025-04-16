def get_labels(self):
    """Loads annotations from a JSON file, filters, and normalizes bounding boxes for each image."""
    labels = []
    LOGGER.info('Loading annotation file...')
    with open(self.json_file, 'r') as f:
        annotations = json.load(f)
    images = {f"{x['id']:d}": x for x in annotations['images']}
    img_to_anns = defaultdict(list)
    for ann in annotations['annotations']:
        img_to_anns[ann['image_id']].append(ann)
    for img_id, anns in TQDM(img_to_anns.items(), desc=
        f'Reading annotations {self.json_file}'):
        img = images[f'{img_id:d}']
        h, w, f = img['height'], img['width'], img['file_name']
        im_file = Path(self.img_path) / f
        if not im_file.exists():
            continue
        self.im_files.append(str(im_file))
        bboxes = []
        cat2id = {}
        texts = []
        for ann in anns:
            if ann['iscrowd']:
                continue
            box = np.array(ann['bbox'], dtype=np.float32)
            box[:2] += box[2:] / 2
            box[[0, 2]] /= float(w)
            box[[1, 3]] /= float(h)
            if box[2] <= 0 or box[3] <= 0:
                continue
            cat_name = ' '.join([img['caption'][t[0]:t[1]] for t in ann[
                'tokens_positive']])
            if cat_name not in cat2id:
                cat2id[cat_name] = len(cat2id)
                texts.append([cat_name])
            cls = cat2id[cat_name]
            box = [cls] + box.tolist()
            if box not in bboxes:
                bboxes.append(box)
        lb = np.array(bboxes, dtype=np.float32) if len(bboxes) else np.zeros((
            0, 5), dtype=np.float32)
        labels.append({'im_file': im_file, 'shape': (h, w), 'cls': lb[:, 0:
            1], 'bboxes': lb[:, 1:], 'normalized': True, 'bbox_format':
            'xywh', 'texts': texts})
    return labels
