def label_loss_items(self, loss_items=None, prefix='train'):
    """
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
    keys = [f'{prefix}/{x}' for x in self.loss_names]
    if loss_items is not None:
        loss_items = [round(float(x), 5) for x in loss_items]
        return dict(zip(keys, loss_items))
    else:
        return keys
