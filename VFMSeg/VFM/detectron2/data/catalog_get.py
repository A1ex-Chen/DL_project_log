def get(self, name):
    """
        Args:
            name (str): name of a dataset (e.g. coco_2014_train).

        Returns:
            Metadata: The :class:`Metadata` instance associated with this name,
            or create an empty one if none is available.
        """
    assert len(name)
    r = super().get(name, None)
    if r is None:
        r = self[name] = Metadata(name=name)
    return r
