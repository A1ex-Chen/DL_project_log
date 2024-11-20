def __init__(self, model, out_indices=(0, 1, 2, 3, 4), out_map=None,
    out_as_dict=False, no_rewrite=False, feature_concat=False,
    flatten_sequential=False, default_hook_type='forward'):
    super(FeatureHookNet, self).__init__()
    assert not torch.jit.is_scripting()
    self.feature_info = _get_feature_info(model, out_indices)
    self.out_as_dict = out_as_dict
    layers = OrderedDict()
    hooks = []
    if no_rewrite:
        assert not flatten_sequential
        if hasattr(model, 'reset_classifier'):
            model.reset_classifier(0)
        layers['body'] = model
        hooks.extend(self.feature_info.get_dicts())
    else:
        modules = _module_list(model, flatten_sequential=flatten_sequential)
        remaining = {f['module']: (f['hook_type'] if 'hook_type' in f else
            default_hook_type) for f in self.feature_info.get_dicts()}
        for new_name, old_name, module in modules:
            layers[new_name] = module
            for fn, fm in module.named_modules(prefix=old_name):
                if fn in remaining:
                    hooks.append(dict(module=fn, hook_type=remaining[fn]))
                    del remaining[fn]
            if not remaining:
                break
        assert not remaining, f'Return layers ({remaining}) are not present in model'
    self.update(layers)
    self.hooks = FeatureHooks(hooks, model.named_modules(), out_map=out_map)
