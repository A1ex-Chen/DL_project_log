def compute_weight(self, module=None, name=None):
    """
        Computes reparameterized weight value to assign value to module attribute
        with name `name`.
        See WeightNorm class for example.
        Arguments:
            module (nn.Module): module with weight we'd like to reparameterize
        Returns:
            w (Tensor): Tensor object containing value of reparameterized weight
        """
    raise NotImplementedError
