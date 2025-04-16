def compute_comparison_indices_length(distances, dist, max_dist_diff):
    max_idx = len(distances)
    comparisons = []
    for idx, d in enumerate(distances):
        best_idx = -1
        error = max_dist_diff
        for i in range(idx, max_idx):
            if np.abs(distances[i] - (d + dist)) < error:
                best_idx = i
                error = np.abs(distances[i] - (d + dist))
        if best_idx != -1:
            comparisons.append(best_idx)
    return comparisons
