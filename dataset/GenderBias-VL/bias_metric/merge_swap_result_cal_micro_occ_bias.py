def cal_micro_occ_bias(data):
    """
    {
        occ: {
            occ_pairs: [(occ, occ_sim, occtm_bias,occtf_bias,bias), ...]
            measure_bias
        }
    }
    """
    occtm_map = OrderedDict()
    occtf_map = OrderedDict()
    occtm_list = []
    occtf_list = []
    for row in data:
        occ = row['occtm']
        if occ not in occtm_map:
            occtm_map[occ] = {'occ_pairs': [], 'bias': []}
        occtm_map[occ]['occ_pairs'].append(row)
        occtm_map[occ]['bias'].append(row['occtm_bias'])
        occ = row['occtf']
        if occ not in occtf_map:
            occtf_map[occ] = {'occ_pairs': [], 'bias': []}
        occtf_map[occ]['occ_pairs'].append(row)
        occtf_map[occ]['bias'].append(row['occtf_bias'])
    for occ in occtm_map:
        mean_bias = np.mean(occtm_map[occ]['bias'])
        temp = OrderedDict({'occ': occ, 'micro_bias': mean_bias})
        occtm_list.append(temp)
    for occ in occtf_map:
        mean_bias = np.mean(occtf_map[occ]['bias'])
        temp = OrderedDict({'occ': occ, 'micro_bias': mean_bias})
        occtf_list.append(temp)
    occtm_list = sorted(occtm_list, key=lambda x: x['micro_bias'], reverse=True
        )
    occtf_list = sorted(occtf_list, key=lambda x: x['micro_bias'], reverse=
        False)
    return occtm_list, occtf_list
