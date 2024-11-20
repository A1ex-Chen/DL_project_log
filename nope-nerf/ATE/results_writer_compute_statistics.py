def compute_statistics(data_vec):
    stats = dict()
    if len(data_vec) > 0:
        stats['rmse'] = float(np.sqrt(np.dot(data_vec, data_vec) / len(
            data_vec)))
        stats['mean'] = float(np.mean(data_vec))
        stats['median'] = float(np.median(data_vec))
        stats['std'] = float(np.std(data_vec))
        stats['min'] = float(np.min(data_vec))
        stats['max'] = float(np.max(data_vec))
        stats['num_samples'] = int(len(data_vec))
    else:
        stats['rmse'] = 0
        stats['mean'] = 0
        stats['median'] = 0
        stats['std'] = 0
        stats['min'] = 0
        stats['max'] = 0
        stats['num_samples'] = 0
    return stats
