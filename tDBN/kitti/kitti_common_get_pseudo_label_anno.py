def get_pseudo_label_anno():
    annotations = {}
    annotations.update({'name': np.array(['Car']), 'truncated': np.array([
        0.0]), 'occluded': np.array([0]), 'alpha': np.array([0.0]), 'bbox':
        np.array([[0.1, 0.1, 15.0, 15.0]]), 'dimensions': np.array([[0.1, 
        0.1, 15.0, 15.0]]), 'location': np.array([[0.1, 0.1, 15.0]]),
        'rotation_y': np.array([[0.1, 0.1, 15.0]])})
    return annotations
