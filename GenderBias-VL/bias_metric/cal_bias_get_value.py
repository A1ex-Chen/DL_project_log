def get_value(data_map, key):
    tempalate = {'mean_prob_gap': 0, 'prob_gap_list_id': []}
    data = data_map.get(key, None)
    if data is None:
        return tempalate
    else:
        return data
