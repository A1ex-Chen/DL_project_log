def __init__(self, result_file, test_file, cf_result_file, cf_test_file,
    save_dir) ->None:
    self.result_data = json.load(open(result_file, 'rb'))
    self.test_data = json.load(open(test_file, 'rb'))
    self.cf_result_data = json.load(open(cf_result_file, 'rb'))
    self.cf_test_data = json.load(open(cf_test_file, 'rb'))
    assert len(self.result_data) == len(self.test_data
        ), 'result and test data should have the same length'
    assert len(self.cf_result_data) == len(self.cf_test_data
        ), 'cf result and test data should have the same length'
    assert len(self.result_data) == len(self.cf_result_data
        ), 'result and cf result should have the same length'
    similar_occ_path = (
        '../construction/vq_generation/similarity/occ_merge_filter_sim.csv')
    self.similar_occ_data, self.similar_occ_data_map = read_file(
        similar_occ_path)
    self.merge_data = self.analyze(self.test_data, self.result_data, type=
        'base')
    self.base_acc_for_pair = self.get_acc(self.merge_data)
    self.cf_merge_data = self.analyze(self.cf_test_data, self.
        cf_result_data, type='cf')
    self.cf_acc_for_pair = self.get_acc(self.cf_merge_data)
    self.cmp_data, self.cmp_data_ppl = self.merge_occ_base_cf(self.
        merge_data, self.cf_merge_data)
    self.occ_bias_pair_ppl_list = self.cal_occ_bias_probablity(self.
        cmp_data_ppl)
    file_name = 'occ_bias_pair_probablity_difference.csv'
    file_name = os.path.join(save_dir, file_name)
    self.write_csv(file_name, self.occ_bias_pair_ppl_list)
    self.occ_bias_pair_list = self.cal_occ_bias_outcome(self.cmp_data)
    file_name = 'occ_bias_pair_outcome_difference.csv'
    file_name = os.path.join(save_dir, file_name)
    self.write_csv(file_name, self.occ_bias_pair_list)
