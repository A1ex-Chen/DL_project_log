def create_dataset_table(self, dataset, class_to_id, name='dataset'):
    artifact = wandb.Artifact(name=name, type='dataset')
    img_files = tqdm([dataset.path]) if isinstance(dataset.path, str) and Path(
        dataset.path).is_dir() else None
    img_files = tqdm(dataset.img_files) if not img_files else img_files
    for img_file in img_files:
        if Path(img_file).is_dir():
            artifact.add_dir(img_file, name='data/images')
            labels_path = 'labels'.join(dataset.path.rsplit('images', 1))
            artifact.add_dir(labels_path, name='data/labels')
        else:
            artifact.add_file(img_file, name='data/images/' + Path(img_file
                ).name)
            label_file = Path(img2label_paths([img_file])[0])
            artifact.add_file(str(label_file), name='data/labels/' +
                label_file.name) if label_file.exists() else None
    table = wandb.Table(columns=['id', 'train_image', 'Classes', 'name'])
    class_set = wandb.Classes([{'id': id, 'name': name} for id, name in
        class_to_id.items()])
    for si, (img, labels, paths, shapes) in enumerate(tqdm(dataset)):
        height, width = shapes[0]
        labels[:, 2:] = xywh2xyxy(labels[:, 2:].view(-1, 4)) * torch.Tensor([
            width, height, width, height])
        box_data, img_classes = [], {}
        for cls, *xyxy in labels[:, 1:].tolist():
            cls = int(cls)
            box_data.append({'position': {'minX': xyxy[0], 'minY': xyxy[1],
                'maxX': xyxy[2], 'maxY': xyxy[3]}, 'class_id': cls,
                'box_caption': '%s' % class_to_id[cls], 'scores': {'acc': 1
                }, 'domain': 'pixel'})
            img_classes[cls] = class_to_id[cls]
        boxes = {'ground_truth': {'box_data': box_data, 'class_labels':
            class_to_id}}
        table.add_data(si, wandb.Image(paths, classes=class_set, boxes=
            boxes), json.dumps(img_classes), Path(paths).name)
    artifact.add(table, name)
    return artifact
