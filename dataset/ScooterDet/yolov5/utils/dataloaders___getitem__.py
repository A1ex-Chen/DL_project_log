def __getitem__(self, i):
    f, j, fn, im = self.samples[i]
    if self.cache_ram and im is None:
        im = self.samples[i][3] = cv2.imread(f)
    elif self.cache_disk:
        if not fn.exists():
            np.save(fn.as_posix(), cv2.imread(f))
        im = np.load(fn)
    else:
        im = cv2.imread(f)
    if self.album_transforms:
        sample = self.album_transforms(image=cv2.cvtColor(im, cv2.
            COLOR_BGR2RGB))['image']
    else:
        sample = self.torch_transforms(im)
    return sample, j
