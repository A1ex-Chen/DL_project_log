def recursive_find_attn_block(module):
    if hasattr(module, '_from_deprecated_attn_block'
        ) and module._from_deprecated_attn_block:
        deprecated_attention_block_modules.append(module)
    for sub_module in module.children():
        recursive_find_attn_block(sub_module)
