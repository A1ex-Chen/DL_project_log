def encode_image(self, image):
    x = self.visual(image)
    x = x @ self.image_projection
    return x
