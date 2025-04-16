def get_mosaic(self, index, shape):
    """Gets images and labels after mosaic augments"""
    indices = [index] + random.choices(range(0, len(self.img_paths)), k=3)
    random.shuffle(indices)
    imgs, hs, ws, labels = [], [], [], []
    for index in indices:
        img, _, (h, w) = self.load_image(index)
        labels_per_img = self.labels[index]
        imgs.append(img)
        hs.append(h)
        ws.append(w)
        labels.append(labels_per_img)
    img, labels = mosaic_augmentation(shape, imgs, hs, ws, labels, self.hyp,
        self.specific_shape, self.target_height, self.target_width)
    return img, labels
