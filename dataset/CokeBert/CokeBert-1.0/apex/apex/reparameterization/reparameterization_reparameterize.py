def reparameterize(self, name, weight, dim):
    """
        Creates Parameters to be used for reparameterization and creates names that
        for attributes for the module these Parameters will correspond to.
        The parameters will be registered according to the names provided.
        See WeightNorm class for example.
        Arguments:
            module (nn.Module): module with weight we'd like to reparameterize
            name (str, optional): name of weight parameter
            dim (int, optional): dimension over which to compute parameterization
        Returns:
            names (list, str): names of Parameters to be used for reparameterization
            params (list, Parameter): Parameters to be used for reparameterization
        """
    raise NotImplementedError
