def load_data(args):
    """Initialize random data

    Args:
        gParameters: parameters from candle

    Returns:
        train, valid, test sets
    """
    num_classes = args.num_classes
    num_train_samples = args.num_train_samples
    num_valid_samples = args.num_valid_samples
    num_test_samples = args.num_test_samples
    train = MimicDatasetSynthetic(num_docs=num_train_samples, num_classes=
        num_classes)
    valid = MimicDatasetSynthetic(num_docs=num_valid_samples, num_classes=
        num_classes)
    test = MimicDatasetSynthetic(num_docs=num_test_samples, num_classes=
        num_classes)
    return train, valid, test
