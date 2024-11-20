@property
def transforms(self):
    """
        Retrieves the transformations applied to the input data of the loaded model.

        This property returns the transformations if they are defined in the model.

        Returns:
            (object | None): The transform object of the model if available, otherwise None.
        """
    return self.model.transforms if hasattr(self.model, 'transforms') else None
