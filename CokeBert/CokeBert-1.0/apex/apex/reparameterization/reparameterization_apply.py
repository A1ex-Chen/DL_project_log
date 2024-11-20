@staticmethod
def apply(module, name, dim, reparameterization=None, hook_child=True):
    """
        Applies reparametrization to module's `name` parameter and modifies instance attributes as appropriate.
        `hook_child` adds reparameterization hook to direct parent of the parameters. If False, it's added to `module` instead.
        """
    if reparameterization is None:
        reparameterization = Reparameterization
    module2use, name2use = Reparameterization.get_module_and_name(module, name)
    if name2use is None or isinstance(module2use, (torch.nn.Embedding,
        torch.nn.EmbeddingBag)):
        return
    if hook_child:
        fn = reparameterization(name2use, dim, module2use)
    else:
        fn = reparameterization(name, dim, module)
    weight = getattr(module2use, name2use)
    if weight.dim() <= 1:
        return
    del module2use._parameters[name2use]
    names, params = fn.reparameterize(name2use, weight, dim)
    for n, p in zip(names, params):
        module2use.register_parameter(n, p)
    fn.reparameterization_names = names
    setattr(module2use, name2use, None)
    hook_module = module2use
    if not hook_child:
        hook_module = module
    hook_module.register_forward_pre_hook(fn)
    handle = hook_module.register_backward_hook(fn.backward_hook)
    fn.backward_hook_key = handle.id
    return fn
