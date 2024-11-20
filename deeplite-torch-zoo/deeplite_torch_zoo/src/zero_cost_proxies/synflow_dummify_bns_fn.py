def dummify_bns_fn(module):
    if isinstance(module, NORMALIZATION_LAYERS):
        module.forward = lambda x: x
