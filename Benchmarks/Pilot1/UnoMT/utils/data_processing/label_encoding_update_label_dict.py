def update_label_dict(data_root: str, dict_name: str, new_labels: iter):
    """label_dict = update_label_dict('./data/',
                                      'label_dict.txt',
                                      ['some', 'labels'])

    This function will check if there exists dictionary for label encoding.
        * if not, it construct a new encoding dictionary;
        * otherwise, it will load the existing dictionary and update if not
            all he labels to be encoded are in the dictionary;
    Lastly, it returns the updated encoded dictionary.

    For example, the label encoding dictionary for labels ['A', 'B', 'C'] is
    {'A': 0, 'B': 1, 'C': 2}.

    Note that all the labels should be strings.

    Args:
        data_root (str): path to data root folder.
        dict_name (str): label encoding dictionary name.
        new_labels (iter): iterable structure of labels to be encoded.

    Returns:
        dict: update encoding dictionary for labels.
    """
    label_encoding_dict = get_label_dict(data_root=data_root, dict_name=
        dict_name)
    old_labels = [str(line) for line in label_encoding_dict.keys()]
    if len(set(new_labels) - set(old_labels)) != 0:
        logger.debug('Updating encoding dict %s' % dict_name)
        old_idx = len(old_labels)
        for idx, l in enumerate(set(new_labels) - set(old_labels)):
            label_encoding_dict[str(l)] = idx + old_idx
        try:
            os.makedirs(os.path.join(data_root, PROC_FOLDER))
        except FileExistsError:
            pass
        dict_path = os.path.join(data_root, PROC_FOLDER, dict_name)
        with open(dict_path, 'w') as f:
            json.dump(label_encoding_dict, f, indent=4)
    return label_encoding_dict
