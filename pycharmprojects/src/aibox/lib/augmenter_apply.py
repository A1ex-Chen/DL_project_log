def apply(self, image: Tensor, bboxes: Optional[Tensor], mask_image:
    Optional[Tensor], **object_field_dict) ->Tuple:
    bbox_params = A.BboxParams(format='pascal_voc', label_fields=list(
        object_field_dict.keys())) if bboxes is not None else None
    if self.strategy == Augmenter.Strategy.ALL:
        imgaug_augmenter = Sequential(children=self.imgaug_transforms,
            random_order=True)
        albumentations_augmenter = A.Compose(self.albumentations_transforms,
            bbox_params)
    elif self.strategy == Augmenter.Strategy.ONE:
        imgaug_augmenter = OneOf(children=self.imgaug_transforms)
        albumentations_augmenter = A.Compose([A.OneOf(self.
            albumentations_transforms)], bbox_params)
    elif self.strategy == Augmenter.Strategy.SOME:
        imgaug_augmenter = SomeOf(children=self.imgaug_transforms,
            random_order=True)
        albumentations_augmenter = A.Compose([A.OneOf([t]) for t in self.
            albumentations_transforms], bbox_params)
    else:
        raise ValueError('Invalid augmenter strategy')
    image = image.permute(1, 2, 0).mul(255).byte().numpy()
    if bboxes is not None:
        bboxes = bboxes.numpy()
        if mask_image is not None:
            mask_image = mask_image.numpy()
    if bboxes is not None:
        bboxes = BoundingBoxesOnImage([BoundingBox(x1=it[0], y1=it[1], x2=
            it[2], y2=it[3]) for it in bboxes.tolist()], shape=image.shape)
        if mask_image is not None:
            mask_image = SegmentationMapsOnImage(mask_image, shape=image.shape)
    image, bboxes, mask_image = imgaug_augmenter(image=image,
        bounding_boxes=bboxes, segmentation_maps=mask_image)
    if bboxes is not None:
        bboxes = bboxes.clip_out_of_image()
        bboxes = np.array([[bbox.x1, bbox.y1, bbox.x2, bbox.y2] for bbox in
            bboxes]).reshape(-1, 4)
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        kept_indices = (areas >= 1).nonzero()[0]
        bboxes = bboxes[kept_indices]
        if mask_image is not None:
            mask_image = mask_image.get_arr()
            for mask_color in range(1, np.max(mask_image).item() + 1):
                if mask_color - 1 not in kept_indices:
                    mask_image = np.where(mask_image == mask_color, 0,
                        mask_image)
        object_field_dict = {k: [object_field_dict[k][i] for i in
            kept_indices.tolist()] for k in object_field_dict.keys()}
    masks = None
    if bboxes is not None:
        bboxes = bboxes.tolist()
        if mask_image is not None:
            mask_colors = np.arange(1, np.max(mask_image).item() + 1)
            masks = np.tile(mask_image, (mask_colors.shape[0], 1, 1))
            masks = (masks == np.tile(mask_colors.reshape((-1, 1, 1)), (1,
                masks.shape[1], masks.shape[2]))).astype(masks.dtype)
            masks = [it for it in masks]
    aug_dict = albumentations_augmenter(image=image, bboxes=bboxes, masks=
        masks, **object_field_dict)
    image = aug_dict['image']
    bboxes = aug_dict['bboxes']
    masks = aug_dict['masks']
    object_field_dict = {k: aug_dict[k] for k in object_field_dict.keys()}
    if bboxes is not None:
        bboxes = np.array(bboxes).reshape(-1, 4)
        if masks is not None:
            masks = np.stack(masks, axis=0)
            mask_image = masks * np.tile(np.arange(1, masks.shape[0] + 1).
                reshape((-1, 1, 1)), (1, masks.shape[1], masks.shape[2]))
            mask_image = mask_image.astype(masks.dtype)
            mask_image = mask_image.max(axis=0)
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image).float().div(255).permute(2, 0, 1)
    if bboxes is not None:
        bboxes = torch.from_numpy(bboxes).float()
    if mask_image is not None:
        mask_image = torch.from_numpy(mask_image).byte()
    return (image, bboxes, mask_image) + tuple(object_field_dict.values())
