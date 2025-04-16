def weighted_average(metrics: List[Tuple[int, Metrics]]) ->Metrics:
    met = {}
    for i, m in enumerate(metrics):
        met[i] = m[1]['fid']
    return met
