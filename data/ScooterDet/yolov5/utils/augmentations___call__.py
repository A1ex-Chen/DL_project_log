def __call__(self, im):
    im = np.ascontiguousarray(im.transpose((2, 0, 1))[::-1])
    im = torch.from_numpy(im)
    im = im.half() if self.half else im.float()
    im /= 255.0
    return im
