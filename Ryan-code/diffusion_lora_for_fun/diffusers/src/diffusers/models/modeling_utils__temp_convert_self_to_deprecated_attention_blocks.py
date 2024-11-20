def _temp_convert_self_to_deprecated_attention_blocks(self) ->None:
    deprecated_attention_block_modules = []

    def recursive_find_attn_block(module):
        if hasattr(module, '_from_deprecated_attn_block'
            ) and module._from_deprecated_attn_block:
            deprecated_attention_block_modules.append(module)
        for sub_module in module.children():
            recursive_find_attn_block(sub_module)
    recursive_find_attn_block(self)
    for module in deprecated_attention_block_modules:
        module.query = module.to_q
        module.key = module.to_k
        module.value = module.to_v
        module.proj_attn = module.to_out[0]
        del module.to_q
        del module.to_k
        del module.to_v
        del module.to_out
