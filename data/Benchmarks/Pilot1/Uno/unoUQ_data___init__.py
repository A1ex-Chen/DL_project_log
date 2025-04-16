def __init__(self, df_data, indices, target_str, feature_names_list,
    num_features_list, batch_size=32, shuffle=True):
    self.batch_size = batch_size
    index = indices
    if shuffle:
        index = np.random.permutation(index)
    self.index = index
    self.index_cycle = cycle(index)
    self.size = len(index)
    self.steps = np.ceil(self.size / batch_size)
    self.num_features_list = num_features_list
    try:
        target = df_data.columns.get_loc(target_str)
    except KeyError:
        y_fake = np.zeros(df_data.shape[0])
        df_data['fake_target'] = y_fake
        self.target = df_data.columns.get_loc('fake_target')
    else:
        self.target = target
    self.df_data = df_data
    self.offset = self.compute_offset(feature_names_list)
