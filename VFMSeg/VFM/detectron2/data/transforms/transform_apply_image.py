def apply_image(self, img):
    img = Image.fromarray(img)
    return np.asarray(super().apply_image(img))
