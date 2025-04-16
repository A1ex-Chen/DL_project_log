def __iter__(self) ->Iterator[List[np.ndarray]]:
    """
        Yields:
            list[ndarray]: the polygons for one instance.
            Each Tensor is a float64 vector representing a polygon.
        """
    return iter(self.polygons)
