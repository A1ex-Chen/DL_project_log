def __getitem__(self, index):
    """rnaseq, data_src, site, type, category = cl_class_dataset[0]

        Args:
            index (int): index for target data slice.

        Returns:
            tuple: a tuple containing the following five elements:
                * RNA sequence data (np.ndarray of float);
                * one-hot-encoded data source (np.ndarray of float);
                * encoded cell line site (int);
                * encoded cell line type (int);
                * encoded cell line category (int)
        """
    cl_data = self.__cl_array[index]
    rnaseq = np.asarray(cl_data[4], dtype=self.__output_dtype)
    data_src = np.array(cl_data[0], dtype=self.__output_dtype)
    cl_site = np.int64(cl_data[1])
    cl_type = np.int64(cl_data[2])
    cl_category = np.int64(cl_data[3])
    return rnaseq, data_src, cl_site, cl_type, cl_category
