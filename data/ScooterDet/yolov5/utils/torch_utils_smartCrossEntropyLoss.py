def smartCrossEntropyLoss(label_smoothing=0.0):
    if check_version(torch.__version__, '1.10.0'):
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if label_smoothing > 0:
        LOGGER.warning(
            f'WARNING ⚠️ label smoothing {label_smoothing} requires torch>=1.10.0'
            )
    return nn.CrossEntropyLoss()
