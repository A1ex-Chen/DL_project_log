def __getitem__(self, index):
    """Fetching a data sample for a given key.
        This function applies mosaic and mixup augments during training.
        During validation, letterbox augment is applied.
        """
    target_shape = (self.target_height, self.target_width
        ) if self.specific_shape else self.batch_shapes[self.batch_indices[
        index]] if self.rect else self.img_size
    if self.augment and random.random() < self.hyp['mosaic']:
        img, labels = self.get_mosaic(index, target_shape)
        shapes = None
        if random.random() < self.hyp['mixup']:
            img_other, labels_other = self.get_mosaic(random.randint(0, len
                (self.img_paths) - 1), target_shape)
            img, labels = mixup(img, labels, img_other, labels_other)
    else:
        if self.hyp and 'shrink_size' in self.hyp:
            img, (h0, w0), (h, w) = self.load_image(index, self.hyp[
                'shrink_size'])
        else:
            img, (h0, w0), (h, w) = self.load_image(index)
        img, ratio, pad = letterbox(img, target_shape, auto=False, scaleup=
            self.augment)
        shapes = (h0, w0), ((h * ratio / h0, w * ratio / w0), pad)
        labels = self.labels[index].copy()
        if labels.size:
            w *= ratio
            h *= ratio
            boxes = np.copy(labels[:, 1:])
            boxes[:, 0] = w * (labels[:, 1] - labels[:, 3] / 2) + pad[0]
            boxes[:, 1] = h * (labels[:, 2] - labels[:, 4] / 2) + pad[1]
            boxes[:, 2] = w * (labels[:, 1] + labels[:, 3] / 2) + pad[0]
            boxes[:, 3] = h * (labels[:, 2] + labels[:, 4] / 2) + pad[1]
            labels[:, 1:] = boxes
        if self.augment:
            img, labels = random_affine(img, labels, degrees=self.hyp[
                'degrees'], translate=self.hyp['translate'], scale=self.hyp
                ['scale'], shear=self.hyp['shear'], new_shape=target_shape)
    if len(labels):
        h, w = img.shape[:2]
        labels[:, [1, 3]] = labels[:, [1, 3]].clip(0, w - 0.001)
        labels[:, [2, 4]] = labels[:, [2, 4]].clip(0, h - 0.001)
        boxes = np.copy(labels[:, 1:])
        boxes[:, 0] = (labels[:, 1] + labels[:, 3]) / 2 / w
        boxes[:, 1] = (labels[:, 2] + labels[:, 4]) / 2 / h
        boxes[:, 2] = (labels[:, 3] - labels[:, 1]) / w
        boxes[:, 3] = (labels[:, 4] - labels[:, 2]) / h
        labels[:, 1:] = boxes
    if self.augment:
        img, labels = self.general_augment(img, labels)
    labels_out = torch.zeros((len(labels), 6))
    if len(labels):
        labels_out[:, 1:] = torch.from_numpy(labels)
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    return torch.from_numpy(img), labels_out, self.img_paths[index], shapes
