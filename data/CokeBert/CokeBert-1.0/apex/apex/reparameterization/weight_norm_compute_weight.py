def compute_weight(self, module=None, name=None):
    """
        Computes weight normalized weight value to assign value to module attribute
        with name `name`.
        Arguments:
            module (nn.Module): module with weight we'd like to reparameterize
        Returns:
            w (Tensor): Tensor object containing value of reparameterized weight
        """
    if module is None:
        module = self.module
    if name is None:
        name = self.name
    module, name = Reparameterization.get_module_and_name(module, name)
    g = getattr(module, name + '_g')
    v = getattr(module, name + '_v')
    fused_weight_norm = Fused_Weight_Norm.apply
    v = v.contiguous()
    w = fused_weight_norm(v, g, self.dim)
    return w
