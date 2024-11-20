def __init__(self, dataframe: pd.DataFrame, ram_dtype: type=np.float16,
    out_dtype: type=np.float32):
    """dataset = DataFrameDataset(dataframe)

        This function initializes a dataset from given dataframe. Upon init,
        the function will convert dataframe into numpy array for faster
        data retrieval. To save the ram space and make data fetching even
        faster, it will store the data in certain dtype to save some
        unnecessary precision bits.

        However, it will still convert the data into out_dtype during slicing.

        Args:
            dataframe (pd.DataFrame): dataframe for the dataset.
            ram_dtype (type): dtype for data storage in RAM.
            out_dtype (type): dtype for data output during slicing.
        """
    self.__data = dataframe.values.astype(ram_dtype)
    self.__out_dtype = out_dtype
    self.__len = len(self.__data)
