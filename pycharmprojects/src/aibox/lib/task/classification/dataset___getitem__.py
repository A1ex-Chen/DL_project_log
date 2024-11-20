def __getitem__(self, index: int) ->Union[Item, ItemTuple]:
    annotation = self.annotations[index]
    image_id = annotation.image_id
    path_to_image = os.path.join(self._path_to_images_dir, annotation.filename)
    cls = self.category_to_class_dict[annotation.category]
    cls = torch.tensor(cls, dtype=torch.long)
    if self._lmdb_txn is not None:
        binary = self._lmdb_txn.get(annotation.filename.encode())
        with io.BytesIO(binary) as f, Image.open(f) as image:
            image = to_tensor(image)
    else:
        with Image.open(path_to_image).convert('RGB') as image:
            image = to_tensor(image)
    processed_image, process_dict = self.preprocessor.process(image,
        is_train_or_eval=self.mode == self.Mode.TRAIN)
    if self.augmenter is not None:
        processed_image, _, _ = self.augmenter.apply(processed_image,
            bboxes=None, mask_image=None)
    if not self.returns_item_tuple:
        return Dataset.Item(path_to_image, image_id, image, processed_image,
            cls, process_dict)
    else:
        return (path_to_image, image_id, image, processed_image, cls,
            process_dict)
