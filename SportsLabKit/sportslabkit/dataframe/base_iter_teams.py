def iter_teams(self, apply_func=None, drop=True):
    """Iterate over the teams of the dataframe.

        Args:
            apply_func (function, optional): Function to apply to each group. Defaults to None.
            drop (bool, optional): Drop the level of the dataframe. Defaults to True.
        """
    if apply_func is None:

        def apply_func(x):
            return x
    for index, group in self.groupby(level='TeamID', axis=1):
        if drop:
            yield index, apply_func(group.droplevel(level='TeamID', axis=1))
        else:
            yield index, apply_func(group)
