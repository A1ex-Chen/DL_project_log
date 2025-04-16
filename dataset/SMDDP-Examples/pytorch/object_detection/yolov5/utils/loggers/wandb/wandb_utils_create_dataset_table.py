def create_dataset_table(self, dataset: LoadImagesAndLabels, class_to_id:
    Dict[int, str], name: str='dataset'):
    """
        Create and return W&B artifact containing W&B Table of the dataset.

        arguments:
        dataset -- instance of LoadImagesAndLabels class used to iterate over the data to build Table
        class_to_id -- hash map that maps class ids to labels
        name -- name of the artifact

        returns:
        dataset artifact to be logged or used
        """
    artifact = wandb.Artifact(name=name, type='dataset')
    img_files = tqdm([dataset.path]) if isinstance(dataset.path, str) and Path(
        dataset.path).is_dir() else None
    img_files = tqdm(dataset.im_files) if not img_files else img_files
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
        box_data, img_classes = [], {}
        for cls, *xywh in labels[:, 1:].tolist():
            cls = int(cls)
            box_data.append({'position': {'middle': [xywh[0], xywh[1]],
                'width': xywh[2], 'height': xywh[3]}, 'class_id': cls,
                'box_caption': '%s' % class_to_id[cls]})
            img_classes[cls] = class_to_id[cls]
        boxes = {'ground_truth': {'box_data': box_data, 'class_labels':
            class_to_id}}
        table.add_data(si, wandb.Image(paths, classes=class_set, boxes=
            boxes), list(img_classes.values()), Path(paths).name)
    artifact.add(table, name)
    return artifact
