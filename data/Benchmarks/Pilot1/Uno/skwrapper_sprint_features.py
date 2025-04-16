def sprint_features(top_features, n_top=1000):
    str = ''
    for i, feature in enumerate(top_features):
        if i >= n_top:
            break
        str += '{:9.5f}\t{}\n'.format(feature[0], feature[1])
    return str
