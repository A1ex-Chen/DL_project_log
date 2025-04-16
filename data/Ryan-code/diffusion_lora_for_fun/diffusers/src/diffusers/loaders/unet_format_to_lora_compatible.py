def format_to_lora_compatible(key):
    if 'processor' not in key.split('.'):
        return key
    return key.replace('.processor', '').replace('to_out_lora', 'to_out.0.lora'
        ).replace('_lora', '.lora')
