def __getitem__(self, index):
    """rnaseq, drug_feature, concentration, growth = dataset[0]

        This function fetches a single sample of drug response data along
        with the corresponding drug features and RNA sequence.

        Note that all the returned values are in ndarray format with the
        type specified during dataset initialization.

        Args:
            index (int): index for drug response data.

        Returns:
            tuple: a tuple of np.ndarray, with RNA sequence data,
                drug features, concentration, and growth.
        """
    drug_resp = self.__drug_resp_array[index]
    drug_feature = self.__drug_feature_dict[drug_resp[1]]
    rnaseq = self.__rnaseq_dict[drug_resp[2]]
    drug_feature = drug_feature.astype(self.__output_dtype)
    rnaseq = rnaseq.astype(self.__output_dtype)
    concentration = np.array([drug_resp[3]], dtype=self.__output_dtype)
    growth = np.array([drug_resp[4]], dtype=self.__output_dtype)
    return rnaseq, drug_feature, concentration, growth
