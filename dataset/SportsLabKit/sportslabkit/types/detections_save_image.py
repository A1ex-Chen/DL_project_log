def save_image(self, path: (str | Path), **kwargs):
    image = self.show(**kwargs)
    image.save(path)
