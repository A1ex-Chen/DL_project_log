def __getitem__(self, idx):
    image_file, label = self._image_files[idx], self._labels[idx]
    image = PIL.Image.open(image_file).convert('RGB')
    if self._map_to_imagenet_labels:
        label = IMAGEWOOF_IMAGENET_CLS_LABEL_MAP[label]
    if self.transform:
        image = self.transform(image)
    if self.target_transform:
        label = self.target_transform(label)
    return image, label
