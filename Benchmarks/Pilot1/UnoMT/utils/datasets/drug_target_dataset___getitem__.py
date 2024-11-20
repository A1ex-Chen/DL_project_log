def __getitem__(self, index):
    drug_target_data = self.__drug_target_array[index]
    drug_feature = np.asarray(drug_target_data[1], dtype=self.__output_dtype)
    target = np.int64(drug_target_data[0])
    return drug_feature, target
