def get_transform(self, image):
    assert image.shape[-1] == 3, 'RandomLighting only works on RGB images'
    weights = np.random.normal(scale=self.scale, size=3)
    return BlendTransform(src_image=self.eigen_vecs.dot(weights * self.
        eigen_vals), src_weight=1.0, dst_weight=1.0)
