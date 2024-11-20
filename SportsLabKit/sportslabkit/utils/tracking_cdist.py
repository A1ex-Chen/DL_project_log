def cdist(detections, tracks, funcs, reduction='mean'):
    costs = []
    for func in funcs:
        C = np.zeros((len(detections), len(tracks)))
        for di, detection in enumerate(detections):
            for ti, track in enumerate(tracks):
                C[di, ti] = cost(detection, track.detections[-1], func=func)
        costs.append(C)
    if reduction == 'mean':
        return np.mean(costs, axis=0)
    else:
        raise NotImplementedError(f'reduction(`{reduction}`) not implemented')
