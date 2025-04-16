def label_loss_items(self, loss_items=None, prefix='train'):
    """
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
    keys = [f'{prefix}/{x}' for x in self.loss_names]
    if loss_items is None:
        return keys
    loss_items = [round(float(loss_items), 5)]
    return dict(zip(keys, loss_items))
