def __getitem__(self, i):
    """Returns subset of data and targets corresponding to given indices."""
    f, j, fn, im = self.samples[i]
    if self.cache_ram:
        if im is None:
            im = self.samples[i][3] = cv2.imread(f)
    elif self.cache_disk:
        if not fn.exists():
            np.save(fn.as_posix(), cv2.imread(f), allow_pickle=False)
        im = np.load(fn)
    else:
        im = cv2.imread(f)
    im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    sample = self.torch_transforms(im)
    return {'img': sample, 'cls': j}
