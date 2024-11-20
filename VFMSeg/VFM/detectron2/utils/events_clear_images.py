def clear_images(self):
    """
        Delete all the stored images for visualization. This should be called
        after images are written to tensorboard.
        """
    self._vis_data = []
