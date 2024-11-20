def activation_checkpointing_fn(self, module):
    return isinstance(module, MPTBlock)
