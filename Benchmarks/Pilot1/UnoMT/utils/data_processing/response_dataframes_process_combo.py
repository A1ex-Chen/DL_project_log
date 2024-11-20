def process_combo(dict_value: list):
    conc_tuple = dict_value[2]
    grth_tuple = dict_value[3]
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            corr = stats.pearsonr(x=conc_tuple, y=grth_tuple)[0]
        except (Warning, ValueError):
            corr = 0.0
    return [dict_value[0], dict_value[1], len(conc_tuple), np.mean(
        grth_tuple), corr]
