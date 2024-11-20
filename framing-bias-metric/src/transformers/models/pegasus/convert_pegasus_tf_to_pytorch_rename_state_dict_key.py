def rename_state_dict_key(k):
    for pegasus_name, hf_name in PATTERNS:
        k = k.replace(pegasus_name, hf_name)
    return k
