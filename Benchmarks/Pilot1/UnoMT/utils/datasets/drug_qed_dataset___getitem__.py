def __getitem__(self, index):
    drug_qed_data = self.__drug_qed_array[index]
    drug_feature = np.asarray(drug_qed_data[1], dtype=self.__output_dtype)
    qed = np.array([drug_qed_data[0]], dtype=self.__output_dtype)
    return drug_feature, qed
