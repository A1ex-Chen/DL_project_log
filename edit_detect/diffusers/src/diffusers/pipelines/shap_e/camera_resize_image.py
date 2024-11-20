def resize_image(self, width: int, height: int
    ) ->'DifferentiableProjectiveCamera':
    """
        Creates a new camera for the resized view assuming the aspect ratio does not change.
        """
    assert width * self.height == height * self.width, 'The aspect ratio should not change.'
    return DifferentiableProjectiveCamera(origin=self.origin, x=self.x, y=
        self.y, z=self.z, width=width, height=height, x_fov=self.x_fov,
        y_fov=self.y_fov)
