def get_conv(model_name):
    if 'llama-2' in model_name.lower():
        conv_mode = 'llava_llama_2'
    elif 'v1' in model_name.lower():
        conv_mode = 'llava_v1'
    elif 'mpt' in model_name.lower():
        conv_mode = 'mpt'
    else:
        conv_mode = 'llava_v0'
    conv = conv_templates[conv_mode].copy()
    return conv
