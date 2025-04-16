def encode_int_to_onehot(labels: iter, num_classes: int=None):
    """one_hot_labels = encode_int_to_onehot(int_labels, num_classes=10)

    This function converts an iterable structure of integer labels into
    one-hot encoding.

    Args:
        labels (iter): an iterable structure of int labels to be encoded.
        num_classes (int): number of classes for labels. When set to None,
            the function will infer from given labels.

    Returns:
        list: list of one-hot-encoded labels.
    """
    if num_classes is None:
        if len(set(labels)) != np.amax(labels) + 1:
            logger.warning(
                'Possible incomplete labels.Set the num_classes to ensure the correctness.'
                )
        num_classes = len(set(labels))
    else:
        assert num_classes >= len(set(labels))
    encoded_labels = []
    for label in labels:
        encoded = [0] * num_classes
        encoded[label] = 1
        encoded_labels.append(encoded)
    return encoded_labels
