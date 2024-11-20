def __call__(self, x):
    sigma = random.uniform(self.sigma[0], self.sigma[1])
    x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
    return x
