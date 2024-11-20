def is_fused(self, thresh=10):
    """
        Check if the model has less than a certain threshold of BatchNorm layers.

        Args:
            thresh (int, optional): The threshold number of BatchNorm layers. Default is 10.

        Returns:
            (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.
        """
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)
    return sum(isinstance(v, bn) for v in self.modules()) < thresh
