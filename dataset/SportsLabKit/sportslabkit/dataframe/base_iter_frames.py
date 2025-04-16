def iter_frames(self, apply_func=None):
    """Iterate over the frames of the dataframe.

        Args:
            apply_func (function, optional): Function to apply to each group. Defaults to None.
        """
    if apply_func is None:

        def apply_func(x):
            return x
    for index, group in self.groupby('frame'):
        yield index, apply_func(group)
