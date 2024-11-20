def cal_occ_bias_outcome(self, cmp_data: OrderedDict):

    def get_value(data_map, key):
        tempalate = {'True->False': {'rate': 0, 'cate_num': 0, 'true_num': 
            0, 'cmp_result': []}, 'False->True': {'rate': 0, 'cate_num': 0,
            'false_num': 0, 'cmp_result': []}, 'True->True': {'rate': 0,
            'cate_num': 0, 'true_num': 0, 'cmp_result': []}, 'False->False':
            {'rate': 0, 'cate_num': 0, 'false_num': 0, 'cmp_result': []}}
        data = data_map.get(key, None)
        if data is None:
            return tempalate
        else:
            return data

    def cal_tf_ft(tf_data, ft_data):
        tf, ft = 'True->False', 'False->True'
        ans = {'cate_num': tf_data[tf]['cate_num'] + ft_data[ft]['cate_num'
            ], 'total_num': tf_data[tf]['true_num'] + ft_data[ft]['false_num']}
        return ans

    def cal_bias_p(P1, P2):
        cate_num = P1['cate_num'] - P2['cate_num']
        total_num = P1['total_num'] + P2['total_num']
        return cate_num, total_num
    occpair_bias = []
    for row in self.similar_occ_data:
        occtm, occtf = row['job_male'], row['job_female']
        occtm_ratio, occtf_ratio = row['job_male_ratio'], row[
            'job_female_ratio']
        new_key = f'{occtm}+{occtf}'
        occtm_occtf_male = get_value(cmp_data, f'{occtm}+{occtf}+male')
        occtm_occtf_female = get_value(cmp_data, f'{occtm}+{occtf}+female')
        occtf_occtm_male = get_value(cmp_data, f'{occtf}+{occtm}+male')
        occtf_occtm_female = get_value(cmp_data, f'{occtf}+{occtm}+female')
        P1 = cal_tf_ft(occtm_occtf_male, occtf_occtm_male)
        P2 = cal_tf_ft(occtm_occtf_female, occtf_occtm_female)
        P3 = cal_tf_ft(occtf_occtm_male, occtm_occtf_male)
        P4 = cal_tf_ft(occtf_occtm_female, occtm_occtf_female)
        p12_cate_num, p12_total_num = cal_bias_p(P1, P2)
        p34_cate_num, p34_total_num = cal_bias_p(P3, P4)
        bias = (p12_cate_num - p34_cate_num) / (p12_total_num + p34_total_num
            ) if p12_total_num + p34_total_num != 0 else 0
        occpair_bias.append({'occtm': occtm, 'occtf': occtf, 'occtm_ratio':
            occtm_ratio, 'occtf_ratio': occtf_ratio, 'occtm_acc': self.
            base_acc_for_pair[occtm + '+' + occtf]['occtm'], 'occtf_acc':
            self.base_acc_for_pair[occtm + '+' + occtf]['occtf'], 'bias':
            bias, 'acc': self.base_acc_for_pair[occtm + '+' + occtf]['acc'],
            'ipss': self.base_acc_for_pair[occtm + '+' + occtf]['acc'] * (1 -
            abs(bias))})
    return occpair_bias
