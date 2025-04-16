def load_all_ratings(min_ratings=1):
    columns = ['user_id', 'item_id', 'rating', 'rating_timestamp', 'type']
    ratings_data = Rating.objects.all().values(*columns)
    ratings = pd.DataFrame.from_records(ratings_data, columns=columns)
    item_count = ratings[['item_id', 'rating']].groupby('item_id').count()
    item_count = item_count.reset_index()
    item_count['rating'] = item_count['rating'].astype(np.dtype(Decimal))
    item_ids = item_count.loc[item_count['rating'] > min_ratings]['item_id']
    ratings = ratings[ratings['item_id'].isin(item_ids)]
    ratings['rating'] = ratings['rating'].astype(np.dtype(Decimal))
    return ratings
