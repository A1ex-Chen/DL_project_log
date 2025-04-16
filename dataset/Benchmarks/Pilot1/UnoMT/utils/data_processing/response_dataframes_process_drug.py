def process_drug(drug: str, dict_value: list):
    num_cl = dict_value[0]
    records_tuple = dict_value[1]
    assert num_cl == len(records_tuple)
    grth_tuple = dict_value[2]
    corr_tuple = dict_value[3]
    num_rec = np.sum(records_tuple)
    avg_grth = np.average(a=grth_tuple, weights=records_tuple)
    avg_corr = np.average(a=corr_tuple, weights=records_tuple)
    return [drug, num_cl, num_rec, avg_grth, avg_corr]
