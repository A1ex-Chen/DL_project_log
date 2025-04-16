def convert_hf_name_to_opus_name(hf_model_name):
    """
    Relies on the assumption that there are no language codes like pt_br in models that are not in GROUP_TO_OPUS_NAME.
    """
    hf_model_name = remove_prefix(hf_model_name, ORG_NAME)
    if hf_model_name in GROUP_TO_OPUS_NAME:
        opus_w_prefix = GROUP_TO_OPUS_NAME[hf_model_name]
    else:
        opus_w_prefix = hf_model_name.replace('_', '+')
    return remove_prefix(opus_w_prefix, 'opus-mt-')
