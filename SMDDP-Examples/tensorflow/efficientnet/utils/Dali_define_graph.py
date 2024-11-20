def define_graph(self):
    inputs = self.input(name='Reader')
    images = inputs['image/encoded']
    labels = inputs['image/class/label']
    labels -= 1
    labels = self.one_hot(labels).gpu()
    images = self.decode(images)
    if not self.training:
        shapes = self.shapes(images)
        h = self.extract_h(shapes, dali.types.Constant(np.array([0], dtype=
            np.float32)), dali.types.Constant(np.array([1], dtype=np.float32)))
        w = self.extract_w(shapes, dali.types.Constant(np.array([1], dtype=
            np.float32)), dali.types.Constant(np.array([1], dtype=np.float32)))
        CROP_PADDING = 32
        CROP_H = h * h / (h + CROP_PADDING)
        CROP_W = w * w / (w + CROP_PADDING)
        CROP_H = self.cast_float(CROP_H)
        CROP_W = self.cast_float(CROP_W)
        images = images.gpu()
        images = self.crop(images, crop_h=CROP_H, crop_w=CROP_W)
    images = self.resize(images)
    images = self.normalize(images)
    return images, labels
