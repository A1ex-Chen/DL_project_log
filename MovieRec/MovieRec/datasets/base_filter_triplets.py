def filter_triplets(self, df):
    print('Filtering triplets')
    if self.min_sc > 0:
        item_sizes = df.groupby('sid').size()
        good_items = item_sizes.index[item_sizes >= self.min_sc]
        df = df[df['sid'].isin(good_items)]
    if self.min_uc > 0:
        user_sizes = df.groupby('uid').size()
        good_users = user_sizes.index[user_sizes >= self.min_uc]
        df = df[df['uid'].isin(good_users)]
    return df
