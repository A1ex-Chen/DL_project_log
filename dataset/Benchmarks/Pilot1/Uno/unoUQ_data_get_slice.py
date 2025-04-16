def get_slice(self, size=None, contiguous=True):
    size = size or self.size
    index = list(islice(self.index_cycle, size))
    df_orig = self.df_data.iloc[index, :]
    df = df_orig.copy()
    x_list = []
    start = self.offset
    for i, numf in enumerate(self.num_features_list):
        end = start + numf
        mat = df.iloc[:, start:end].values
        if contiguous:
            mat = np.ascontiguousarray(mat)
        x_list.append(mat)
        start = end
    mat = df.iloc[:, self.target].values
    if contiguous:
        mat = np.ascontiguousarray(mat)
    y = mat
    return x_list, y
