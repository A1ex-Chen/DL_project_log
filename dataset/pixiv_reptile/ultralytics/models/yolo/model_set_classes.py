def set_classes(self, classes):
    """
        Set classes.

        Args:
            classes (List(str)): A list of categories i.e. ["person"].
        """
    self.model.set_classes(classes)
    background = ' '
    if background in classes:
        classes.remove(background)
    self.model.names = classes
    if self.predictor:
        self.predictor.model.names = classes
