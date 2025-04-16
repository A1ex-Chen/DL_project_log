def set_image(self, image: Image.Image):
    assert self.image is None, f'{image}'
    self.image = image
