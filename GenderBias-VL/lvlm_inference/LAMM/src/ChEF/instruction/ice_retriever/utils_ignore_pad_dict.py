def ignore_pad_dict(features):
    res_dict = {}
    if 'metadata' in features[0]:
        res_dict['metadata'] = ListWrapper([x.pop('metadata') for x in
            features])
    return res_dict
