def _mix_transform(self, labels):
    """Applies MixUp augmentation as per https://arxiv.org/pdf/1710.09412.pdf."""
    r = np.random.beta(32.0, 32.0)
    labels2 = labels['mix_labels'][0]
    labels['img'] = (labels['img'] * r + labels2['img'] * (1 - r)).astype(np
        .uint8)
    labels['instances'] = Instances.concatenate([labels['instances'],
        labels2['instances']], axis=0)
    labels['cls'] = np.concatenate([labels['cls'], labels2['cls']], 0)
    return labels
