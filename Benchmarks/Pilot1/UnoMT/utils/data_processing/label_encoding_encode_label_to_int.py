def encode_label_to_int(data_root: str, dict_name: str, labels: iter):
    """encoded_labels = label_encoding('./data/',
                                       'label_dict.txt',
                                       dataframe['column'])

    This function encodes a iterable structure of labels into list of integer.

    Args:
        data_root (str): path to data root folder.
        dict_name (str): label encoding dictionary name.
        labels (iter): an iterable structure of labels to be encoded.

    Returns:
        list: list of integer encoded labels.
    """
    label_encoding_dict = update_label_dict(data_root=data_root, dict_name=
        dict_name, new_labels=labels)
    return [label_encoding_dict[str(s)] for s in labels]
