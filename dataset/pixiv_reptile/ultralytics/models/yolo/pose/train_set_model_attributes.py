def set_model_attributes(self):
    """Sets keypoints shape attribute of PoseModel."""
    super().set_model_attributes()
    self.model.kpt_shape = self.data['kpt_shape']
