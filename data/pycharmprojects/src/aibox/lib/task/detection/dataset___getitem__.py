def __getitem__(self, index: int) ->Union[Item, ItemTuple]:
    annotation = self.annotations[index]
    image_id = annotation.image_id
    path_to_image = os.path.join(self._path_to_images_dir, annotation.filename)
    bboxes = [obj.bbox.tolist() for obj in annotation.objects]
    bboxes = torch.tensor(bboxes, dtype=torch.float).reshape(-1, 4)
    classes = [self.category_to_class_dict[obj.name] for obj in annotation.
        objects]
    difficulties = [obj.difficulty for obj in annotation.objects]
    if self._lmdb_txn is not None:
        binary = self._lmdb_txn.get(annotation.filename.encode())
        with io.BytesIO(binary) as f, Image.open(f) as image:
            image = to_tensor(image)
    else:
        with Image.open(path_to_image).convert('RGB') as image:
            image = to_tensor(image)
    processed_image, process_dict = self.preprocessor.process(image,
        is_train_or_eval=self.mode == self.Mode.TRAIN)
    processed_bboxes = bboxes.clone()
    processed_bboxes[:, [0, 2]] *= process_dict[Preprocessor.
        PROCESS_KEY_WIDTH_SCALE]
    processed_bboxes[:, [1, 3]] *= process_dict[Preprocessor.
        PROCESS_KEY_HEIGHT_SCALE]
    if self.augmenter is not None:
        processed_image, processed_bboxes, _, classes, difficulties = (self
            .augmenter.apply(processed_image, processed_bboxes, mask_image=
            None, classes=classes, difficulties=difficulties))
    assert len(processed_bboxes) == len(classes) == len(difficulties)
    classes = torch.tensor(classes, dtype=torch.long)
    difficulties = torch.tensor(difficulties, dtype=torch.int8)
    if not self.returns_item_tuple:
        return Dataset.Item(path_to_image, image_id, image, processed_image,
            bboxes, processed_bboxes, classes, difficulties, process_dict)
    else:
        return (path_to_image, image_id, image, processed_image, bboxes,
            processed_bboxes, classes, difficulties, process_dict)
