def fsdp_wrap_fn(self, module):
    return isinstance(module, MPTBlock)
