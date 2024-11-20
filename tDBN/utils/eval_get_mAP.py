def get_mAP(prec):
    sums = 0
    for i in range(0, len(prec), 4):
        sums += prec[i]
    return sums / 11 * 100
