def _undo_temp_convert_self_to_deprecated_attention_blocks(self) ->None:
    deprecated_attention_block_modules = []

    def recursive_find_attn_block(module) ->None:
        if hasattr(module, '_from_deprecated_attn_block'
            ) and module._from_deprecated_attn_block:
            deprecated_attention_block_modules.append(module)
        for sub_module in module.children():
            recursive_find_attn_block(sub_module)
    recursive_find_attn_block(self)
    for module in deprecated_attention_block_modules:
        module.to_q = module.query
        module.to_k = module.key
        module.to_v = module.value
        module.to_out = nn.ModuleList([module.proj_attn, nn.Dropout(module.
            dropout)])
        del module.query
        del module.key
        del module.value
        del module.proj_attn
