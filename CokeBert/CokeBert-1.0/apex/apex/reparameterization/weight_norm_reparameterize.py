def reparameterize(self, name, weight, dim):
    """
        Creates Parameters v and gto be used for weight normalization
        and creates names that for attributes for the module these Parameters
        will correspond to. The parameters will be registered according to the names
        provided.
        Arguments:
            module (nn.Module): module with weight we'd like to reparameterize
            name (str, optional): name of weight parameter
            dim (int, optional): dimension over which to compute parameterization
        Returns:
            names (list, str): names of Parameters to be used for reparameterization
            params (list, Parameter): Parameters to be used for reparameterization
        """
    names = [name + '_g', name + '_v']
    params = [Parameter(_norm(weight, dim).data), Parameter(weight.data)]
    return names, params
