def __getitem__(self, index):
    index = self.indices[index]
    hyp = self.hyp
    mosaic = self.mosaic and random.random() < hyp['mosaic']
    masks = []
    if mosaic:
        img, labels, segments = self.load_mosaic(index)
        shapes = None
        if random.random() < hyp['mixup']:
            img, labels, segments = mixup(img, labels, segments, *self.
                load_mosaic(random.randint(0, self.n - 1)))
    else:
        img, (h0, w0), (h, w) = self.load_image(index)
        shape = self.batch_shapes[self.batch[index]
            ] if self.rect else self.img_size
        img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.
            augment)
        shapes = (h0, w0), ((h / h0, w / w0), pad)
        labels = self.labels[index].copy()
        segments = self.segments[index].copy()
        if len(segments):
            for i_s in range(len(segments)):
                segments[i_s] = xyn2xy(segments[i_s], ratio[0] * w, ratio[1
                    ] * h, padw=pad[0], padh=pad[1])
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1
                ] * h, padw=pad[0], padh=pad[1])
        if self.augment:
            img, labels, segments = random_perspective(img, labels,
                segments=segments, degrees=hyp['degrees'], translate=hyp[
                'translate'], scale=hyp['scale'], shear=hyp['shear'],
                perspective=hyp['perspective'])
    nl = len(labels)
    if nl:
        labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.
            shape[0], clip=True, eps=0.001)
        if self.overlap:
            masks, sorted_idx = polygons2masks_overlap(img.shape[:2],
                segments, downsample_ratio=self.downsample_ratio)
            masks = masks[None]
            labels = labels[sorted_idx]
        else:
            masks = polygons2masks(img.shape[:2], segments, color=1,
                downsample_ratio=self.downsample_ratio)
    masks = torch.from_numpy(masks) if len(masks) else torch.zeros(1 if
        self.overlap else nl, img.shape[0] // self.downsample_ratio, img.
        shape[1] // self.downsample_ratio)
    if self.augment:
        img, labels = self.albumentations(img, labels)
        nl = len(labels)
        augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp[
            'hsv_v'])
        if random.random() < hyp['flipud']:
            img = np.flipud(img)
            if nl:
                labels[:, 2] = 1 - labels[:, 2]
                masks = torch.flip(masks, dims=[1])
        if random.random() < hyp['fliplr']:
            img = np.fliplr(img)
            if nl:
                labels[:, 1] = 1 - labels[:, 1]
                masks = torch.flip(masks, dims=[2])
    labels_out = torch.zeros((nl, 6))
    if nl:
        labels_out[:, 1:] = torch.from_numpy(labels)
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    return torch.from_numpy(img), labels_out, self.im_files[index
        ], shapes, masks
