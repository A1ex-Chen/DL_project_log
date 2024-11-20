def get_user_item_matrix(df):
    df = df[['user_id', 'item_id', 'rating']]
    df_rating = df.groupby(['user_id', 'item_id'])['rating'].mean()
    df_rating = df_rating.unstack()
    df_rating = df_rating.fillna(0)
    return df_rating
