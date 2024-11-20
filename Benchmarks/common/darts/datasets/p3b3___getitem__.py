def __getitem__(self, idx):
    """
        Parameters
        ----------
        index : int
          Index of the data to be loaded.

        Returns
        -------
        (document, target) : tuple
           where target is index of the target class.
        """
    document = self.data[idx]
    if self.transform is not None:
        document = self.transform(document)
    targets = {}
    for key, value in self.targets.items():
        subset = value[idx]
        if self.target_transform is not None:
            subset = self.target_transform(subset)
        targets[key] = subset
    return document, targets
