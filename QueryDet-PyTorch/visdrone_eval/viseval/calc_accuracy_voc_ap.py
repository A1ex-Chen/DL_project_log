def voc_ap(rec, prec):
    mrec = np.concatenate(([0], rec, [1]))
    mpre = np.concatenate(([0], prec, [0]))
    for i in reversed(range(0, len(mpre) - 1)):
        mpre[i] = max(mpre[i], mpre[i + 1])
    i = np.flatnonzero(mrec[1:] != mrec[:-1]) + 1
    ap = np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap
