def dfToDict(df):
    df_ = df.groupby(['user_id'])['item_id'].unique()
    df_ = df_.reset_index()
    dic = df_.to_dict()
    return dic['item_id']
