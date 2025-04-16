def empty_result_anno():
    annotations = {}
    annotations.update({'name': np.array([]), 'truncated': np.array([]),
        'occluded': np.array([]), 'alpha': np.array([]), 'bbox': np.zeros([
        0, 4]), 'dimensions': np.zeros([0, 3]), 'location': np.zeros([0, 3]
        ), 'rotation_y': np.array([]), 'score': np.array([])})
    return annotations
