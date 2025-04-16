def split_df(self, df, user_count):
    if self.args.split == 'leave_one_out':
        print('Splitting')
        user_group = df.groupby('uid')
        user2items = user_group.progress_apply(lambda d: list(d.sort_values
            (by='timestamp')['sid']))
        train, val, test = {}, {}, {}
        for user in range(user_count):
            items = user2items[user]
            train[user], val[user], test[user] = items[:-2], items[-2:-1
                ], items[-1:]
        return train, val, test
    elif self.args.split == 'holdout':
        print('Splitting')
        np.random.seed(self.args.dataset_split_seed)
        eval_set_size = self.args.eval_set_size
        permuted_index = np.random.permutation(user_count)
        train_user_index = permuted_index[:-2 * eval_set_size]
        val_user_index = permuted_index[-2 * eval_set_size:-eval_set_size]
        test_user_index = permuted_index[-eval_set_size:]
        train_df = df.loc[df['uid'].isin(train_user_index)]
        val_df = df.loc[df['uid'].isin(val_user_index)]
        test_df = df.loc[df['uid'].isin(test_user_index)]
        train = dict(train_df.groupby('uid').progress_apply(lambda d: list(
            d['sid'])))
        val = dict(val_df.groupby('uid').progress_apply(lambda d: list(d[
            'sid'])))
        test = dict(test_df.groupby('uid').progress_apply(lambda d: list(d[
            'sid'])))
        return train, val, test
    else:
        raise NotImplementedError
