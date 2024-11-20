@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < len(scores) - 1:
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if r_recall - current_recall < current_recall - l_recall and i < len(
            scores) - 1:
            continue
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    return thresholds
