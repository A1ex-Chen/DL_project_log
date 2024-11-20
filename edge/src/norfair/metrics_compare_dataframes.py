def compare_dataframes(gts, ts):
    """Builds accumulator for each sequence."""
    accs = []
    names = []
    for k, tsacc in ts.items():
        print('Comparing ', k, '...')
        if k in gts:
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc,
                'iou', distth=0.5))
            names.append(k)
    return accs, names
