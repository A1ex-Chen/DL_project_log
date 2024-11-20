def cal_occ_bias_probablity(self, cmp_data: OrderedDict):
    occpair_bias = []

    def get_value(data_map, key):
        tempalate = {'mean_prob_gap': 0, 'prob_gap_list_id': []}
        data = data_map.get(key, None)
        if data is None:
            return tempalate
        else:
            return data
    for row in self.similar_occ_data:
        occtm, occtf = row['job_male'], row['job_female']
        occtm_ratio, occtf_ratio = row['job_male_ratio'], row[
            'job_female_ratio']
        similarity = float(row['similarity'])
        occtm_male = get_value(cmp_data, f'{occtm}+{occtf}+male')
        occtm_female = get_value(cmp_data, f'{occtm}+{occtf}+female')
        occtf_male = get_value(cmp_data, f'{occtf}+{occtm}+male')
        occtf_female = get_value(cmp_data, f'{occtf}+{occtm}+female')
        occtm_bias = occtm_male['mean_prob_gap'] * len(occtm_male[
            'prob_gap_list_id']) - occtm_female['mean_prob_gap'] * len(
            occtm_female['prob_gap_list_id'])
        occtm_bias = occtm_bias / (len(occtm_male['prob_gap_list_id']) +
            len(occtm_female['prob_gap_list_id']))
        occtf_bias = occtf_male['mean_prob_gap'] * len(occtf_male[
            'prob_gap_list_id']) - occtf_female['mean_prob_gap'] * len(
            occtf_female['prob_gap_list_id'])
        occtf_bias = occtf_bias / (len(occtf_male['prob_gap_list_id']) +
            len(occtf_female['prob_gap_list_id']))
        bias = 0.5 * (occtm_bias - occtf_bias)
        occpair_bias.append({'occtm': occtm, 'occtf': occtf, 'occtm_ratio':
            occtm_ratio, 'occtf_ratio': occtf_ratio, 'similarity':
            similarity, 'occtm_bias': occtm_bias, 'occtf_bias': occtf_bias,
            'occtm_acc': self.base_acc_for_pair[occtm + '+' + occtf][
            'occtm'], 'occtf_acc': self.base_acc_for_pair[occtm + '+' +
            occtf]['occtf'], 'bias': bias, 'acc': self.base_acc_for_pair[
            occtm + '+' + occtf]['acc'], 'ipss': self.base_acc_for_pair[
            occtm + '+' + occtf]['acc'] * (1 - abs(bias))})
    return occpair_bias
