def encode_image(self, image):
    return self.visual(image.type(self.dtype))
