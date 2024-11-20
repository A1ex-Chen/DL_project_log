def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    class_counts = np.array([np.bincount(x[:, 0].astype(int), minlength=nc) for
        x in labels])
    return (class_weights.reshape(1, nc) * class_counts).sum(1)
