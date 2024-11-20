def compute_offset(self, feature_names):
    offset = self.df_data.shape[1]
    for name in feature_names:
        col_indices = find_columns_with_str(self.df_data, name)
        if len(col_indices) > 0:
            first_col = np.min(col_indices)
            if first_col < offset:
                offset = first_col
    if offset == self.df_data.shape[1]:
        raise Exception(
            'ERROR ! Feature names from model are not in file. ' +
            'These are features in model: ' + str(sorted(feature_names)) +
            '... Exiting')
    return offset
