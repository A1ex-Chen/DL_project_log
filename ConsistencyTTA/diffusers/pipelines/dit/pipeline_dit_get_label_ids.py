def get_label_ids(self, label: Union[str, List[str]]) ->List[int]:
    """

        Map label strings, *e.g.* from ImageNet, to corresponding class ids.

        Parameters:
            label (`str` or `dict` of `str`): label strings to be mapped to class ids.

        Returns:
            `list` of `int`: Class ids to be processed by pipeline.
        """
    if not isinstance(label, list):
        label = list(label)
    for l in label:
        if l not in self.labels:
            raise ValueError(
                f"""{l} does not exist. Please make sure to select one of the following labels: 
 {self.labels}."""
                )
    return [self.labels[l] for l in label]
