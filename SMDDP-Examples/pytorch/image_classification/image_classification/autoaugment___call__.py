def __call__(self, img):
    if random.random() < self.p1:
        img = self.operation1(img)
    if random.random() < self.p2:
        img = self.operation2(img)
    return img
