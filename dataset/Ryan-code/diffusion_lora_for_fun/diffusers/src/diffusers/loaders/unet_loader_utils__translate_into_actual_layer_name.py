def _translate_into_actual_layer_name(name):
    """Translate user-friendly name (e.g. 'mid') into actual layer name (e.g. 'mid_block.attentions.0')"""
    if name == 'mid':
        return 'mid_block.attentions.0'
    updown, block, attn = name.split('.')
    updown = updown.replace('down', 'down_blocks').replace('up', 'up_blocks')
    block = block.replace('block_', '')
    attn = 'attentions.' + attn
    return '.'.join((updown, block, attn))
