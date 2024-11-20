def __getitem__(self, index: int) ->Union[Item, ItemTuple]:
    annotation = self.annotations[index]
    image_id = annotation.image_id
    path_to_image = os.path.join(self._path_to_images_dir, annotation.filename)
    mask_colors = sorted([obj.mask_color for obj in annotation.objects if 
        obj.mask_color != 0])
    assert mask_colors == list(set(mask_colors))
    path_to_mask_image = os.path.join(self._path_to_segmentations_dir,
        f'{annotation.image_id}.png')
    mask_image = Image.open(path_to_mask_image)
    mask_image = np.array(mask_image, dtype=np.uint8)
    mask_image = torch.from_numpy(mask_image)
    bboxes = self._mask_image_to_bboxes(mask_image, mask_colors)
    masks = self._mask_image_to_masks(mask_image, mask_colors)
    classes = [self.category_to_class_dict[obj.name] for obj in annotation.
        objects if obj.name != 'background']
    difficulties = [obj.difficulty for obj in annotation.objects if obj.
        name != 'background']
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
    processed_mask_image = F.interpolate(input=mask_image.unsqueeze(dim=0).
        unsqueeze(dim=0).float(), scale_factor=(process_dict[Preprocessor.
        PROCESS_KEY_HEIGHT_SCALE], process_dict[Preprocessor.
        PROCESS_KEY_WIDTH_SCALE]), mode='nearest', recompute_scale_factor=True
        ).squeeze(dim=0).squeeze(dim=0).type_as(mask_image)
    processed_mask_image = F.pad(input=processed_mask_image, pad=[0,
        process_dict[Preprocessor.PROCESS_KEY_RIGHT_PAD], 0, process_dict[
        Preprocessor.PROCESS_KEY_BOTTOM_PAD]])
    if self.augmenter is not None:
        (processed_image, processed_bboxes, processed_mask_image, classes,
            difficulties, mask_colors) = (self.augmenter.apply(
            processed_image, processed_bboxes, processed_mask_image,
            classes=classes, difficulties=difficulties, mask_colors=
            mask_colors))
    processed_masks = self._mask_image_to_masks(processed_mask_image,
        mask_colors)
    assert len(processed_bboxes) == len(processed_masks) == len(classes
        ) == len(difficulties)
    if processed_masks.shape[1] != processed_image.shape[1
        ] or processed_masks.shape[2] != processed_image.shape[2]:
        if abs(processed_masks.shape[1] - processed_image.shape[1]
            ) <= 1 and abs(processed_masks.shape[2] - processed_image.shape[2]
            ) <= 1:
            processed_masks = F.interpolate(input=processed_masks.unsqueeze
                (dim=0).float(), size=(processed_image.shape[1],
                processed_image.shape[2]), mode='nearest').squeeze(dim=0
                ).type_as(processed_masks)
        else:
            raise ValueError
    classes = torch.tensor(classes, dtype=torch.long)
    difficulties = torch.tensor(difficulties, dtype=torch.int8)
    if not self.returns_item_tuple:
        return Dataset.Item(path_to_image, image_id, image, processed_image,
            bboxes, processed_bboxes, masks, processed_masks, classes,
            difficulties, process_dict)
    else:
        return (path_to_image, image_id, image, processed_image, bboxes,
            processed_bboxes, masks, processed_masks, classes, difficulties,
            process_dict)
