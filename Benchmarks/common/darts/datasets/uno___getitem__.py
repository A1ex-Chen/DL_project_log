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
    data = self.data['gene_data'][idx]
    if self.transform is not None:
        data = self.transform(data)
    targets = {}
    for key, value in self.targets.items():
        subset = value[idx]
        if self.target_transform is not None:
            subset = self.target_transform(subset)
        targets[key] = subset
    return data, targets
