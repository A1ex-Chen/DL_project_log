def get_acc(self, merge_data):

    def get_value(data_map, key):
        data = data_map.get(key, None)
        if data is None:
            return {'acc': 0, 'acc_num': 0, 'total_num': 0}
        else:
            return data
    output = OrderedDict()
    output_for_pair = OrderedDict()
    for key, value in merge_data.items():
        acc = np.mean([x['metric_result'] for x in value])
        acc_num = np.sum([x['metric_result'] for x in value])
        acc = float(acc)
        acc_num = int(acc_num)
        total_num = len(value)
        output[key] = {'acc': acc, 'acc_num': acc_num, 'total_num': total_num}
    for row in self.similar_occ_data:
        occtm, occtf = row['job_male'], row['job_female']
        new_key = f'{occtm}+{occtf}'
        occtm_occtf_male = get_value(output, f'{occtm}+{occtf}+male')
        occtm_occtf_female = get_value(output, f'{occtm}+{occtf}+female')
        occtf_occtm_male = get_value(output, f'{occtf}+{occtm}+male')
        occtf_occtm_female = get_value(output, f'{occtf}+{occtm}+female')
        occtm_occtf_acc = (occtm_occtf_male['acc_num'] + occtm_occtf_female
            ['acc_num']) / (occtm_occtf_male['total_num'] +
            occtm_occtf_female['total_num'])
        occtf_occtm_acc = (occtf_occtm_male['acc_num'] + occtf_occtm_female
            ['acc_num']) / (occtf_occtm_male['total_num'] +
            occtf_occtm_female['total_num'])
        acc_num = occtm_occtf_male['acc_num'] + occtm_occtf_female['acc_num'
            ] + occtf_occtm_male['acc_num'] + occtf_occtm_female['acc_num']
        total_num = occtm_occtf_male['total_num'] + occtm_occtf_female[
            'total_num'] + occtf_occtm_male['total_num'] + occtf_occtm_female[
            'total_num']
        output_for_pair[new_key] = {'acc': 0.5 * (occtm_occtf_acc +
            occtf_occtm_acc), 'occtm': occtm_occtf_acc, 'occtf':
            occtf_occtm_acc, 'acc_num': acc_num, 'total_num': total_num}
    return output_for_pair
