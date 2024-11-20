def __call__(self, img):
    if self.isPIL:
        img = np.array(img)
    ops = self.get_random_ops()
    for name, prob, level in ops:
        if np.random.random() > prob:
            continue
        args = arg_dict[name](level)
        img = func_dict[name](img, *args)
    return img
