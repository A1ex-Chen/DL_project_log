def preprocess(self, image):
    if len(image.shape) == 3:
        padded_img = np.ones((self.input_size[0], self.input_size[1], 3)
            ) * 114.0
    else:
        padded_img = np.ones(self.input_size) * 114.0
    img = np.array(image)
    r = min(self.input_size[0] / img.shape[0], self.input_size[1] / img.
        shape[1])
    resized_img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] *
        r)), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    padded_img[:int(img.shape[0] * r), :int(img.shape[1] * r)] = resized_img
    image = padded_img
    image = image.astype(np.float32)
    return image, r
