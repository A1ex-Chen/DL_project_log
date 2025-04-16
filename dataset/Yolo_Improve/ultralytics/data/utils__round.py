def _round(labels):
    """Update labels to integer class and 4 decimal place floats."""
    if self.task == 'detect':
        coordinates = labels['bboxes']
    elif self.task in {'segment', 'obb'}:
        coordinates = [x.flatten() for x in labels['segments']]
    elif self.task == 'pose':
        n, nk, nd = labels['keypoints'].shape
        coordinates = np.concatenate((labels['bboxes'], labels['keypoints']
            .reshape(n, nk * nd)), 1)
    else:
        raise ValueError(f'Undefined dataset task={self.task}.')
    zipped = zip(labels['cls'], coordinates)
    return [[int(c[0]), *(round(float(x), 4) for x in points)] for c,
        points in zipped]
