def filter_camera(self, camera):
    assert camera in self.cameras
    self._camera = camera
    return self
