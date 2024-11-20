def labels_to_class_weights(labels, nc=80):
    if labels[0] is None:
        return torch.Tensor()
    labels = np.concatenate(labels, 0)
    classes = labels[:, 0].astype(int)
    weights = np.bincount(classes, minlength=nc)
    weights[weights == 0] = 1
    weights = 1 / weights
    weights /= weights.sum()
    return torch.from_numpy(weights).float()
