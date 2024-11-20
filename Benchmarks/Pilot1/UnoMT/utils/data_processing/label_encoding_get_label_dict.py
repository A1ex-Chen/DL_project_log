def get_label_dict(data_root: str, dict_name: str):
    """label_dict = get_label_list('./data/', 'label_dict.txt')

    Get the encoding dictionary from the given data path.

    Args:
        data_root (str): path to data root folder.
        dict_name (str): label encoding dictionary name.

    Returns:
        dict: encoding dictionary. {} if the dictionary does not exist.
    """
    dict_path = os.path.join(data_root, PROC_FOLDER, dict_name)
    if os.path.exists(dict_path):
        with open(dict_path, 'r') as f:
            label_encoding_dict = json.load(f)
        return label_encoding_dict
    else:
        return {}
