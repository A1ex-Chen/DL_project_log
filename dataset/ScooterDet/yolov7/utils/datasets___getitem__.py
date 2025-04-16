def __getitem__(self, index):
    index = self.indices[index]
    hyp = self.hyp
    mosaic = self.mosaic and random.random() < hyp['mosaic']
    if mosaic:
        if random.random() < 0.8:
            img, labels = load_mosaic(self, index)
        else:
            img, labels = load_mosaic9(self, index)
        shapes = None
        if random.random() < hyp['mixup']:
            if random.random() < 0.8:
                img2, labels2 = load_mosaic(self, random.randint(0, len(
                    self.labels) - 1))
            else:
                img2, labels2 = load_mosaic9(self, random.randint(0, len(
                    self.labels) - 1))
            r = np.random.beta(8.0, 8.0)
            img = (img * r + img2 * (1 - r)).astype(np.uint8)
            labels = np.concatenate((labels, labels2), 0)
    else:
        img, (h0, w0), (h, w) = load_image(self, index)
        shape = self.batch_shapes[self.batch[index]
            ] if self.rect else self.img_size
        img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.
            augment)
        shapes = (h0, w0), ((h / h0, w / w0), pad)
        labels = self.labels[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1
                ] * h, padw=pad[0], padh=pad[1])
    if self.augment:
        if not mosaic:
            img, labels = random_perspective(img, labels, degrees=hyp[
                'degrees'], translate=hyp['translate'], scale=hyp['scale'],
                shear=hyp['shear'], perspective=hyp['perspective'])
        augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp[
            'hsv_v'])
        if random.random() < hyp['paste_in']:
            sample_labels, sample_images, sample_masks = [], [], []
            while len(sample_labels) < 30:
                sample_labels_, sample_images_, sample_masks_ = load_samples(
                    self, random.randint(0, len(self.labels) - 1))
                sample_labels += sample_labels_
                sample_images += sample_images_
                sample_masks += sample_masks_
                if len(sample_labels) == 0:
                    break
            labels = pastein(img, labels, sample_labels, sample_images,
                sample_masks)
    nL = len(labels)
    if nL:
        labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])
        labels[:, [2, 4]] /= img.shape[0]
        labels[:, [1, 3]] /= img.shape[1]
    if self.augment:
        if random.random() < hyp['flipud']:
            img = np.flipud(img)
            if nL:
                labels[:, 2] = 1 - labels[:, 2]
        if random.random() < hyp['fliplr']:
            img = np.fliplr(img)
            if nL:
                labels[:, 1] = 1 - labels[:, 1]
    labels_out = torch.zeros((nL, 6))
    if nL:
        labels_out[:, 1:] = torch.from_numpy(labels)
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    return torch.from_numpy(img), labels_out, self.img_files[index], shapes
