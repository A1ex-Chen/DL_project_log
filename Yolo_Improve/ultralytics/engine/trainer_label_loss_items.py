def label_loss_items(self, loss_items=None, prefix='train'):
    """
        Returns a loss dict with labelled training loss items tensor.

        Note:
            This is not needed for classification but necessary for segmentation & detection
        """
    return {'loss': loss_items} if loss_items is not None else ['loss']
