def convert_opus_name_to_hf_name(x):
    """For OPUS-MT-Train/ DEPRECATED"""
    for substr, grp_name in GROUPS:
        x = x.replace(substr, grp_name)
    return x.replace('+', '_')
