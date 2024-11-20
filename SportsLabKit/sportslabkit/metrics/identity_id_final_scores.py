def id_final_scores(res):
    res['IDR'] = res['IDTP'] / np.maximum(1.0, res['IDTP'] + res['IDFN'])
    res['IDP'] = res['IDTP'] / np.maximum(1.0, res['IDTP'] + res['IDFP'])
    res['IDF1'] = res['IDTP'] / np.maximum(1.0, res['IDTP'] + 0.5 * res[
        'IDFP'] + 0.5 * res['IDFN'])
