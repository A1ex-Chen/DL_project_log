def __call__(self, img):
    img = self.transform(img)
    return img
