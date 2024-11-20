def __call__(self, img):
    if random.random() < self.p1:
        img = self.operation1(img, self.magnitude1)
    if random.random() < self.p2:
        img = self.operation2(img, self.magnitude2)
    return img
