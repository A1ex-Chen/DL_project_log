def load_ratings_df(self):
    folder_path = self._get_rawdata_folder_path()
    file_path = folder_path.joinpath('ratings.csv')
    df = pd.read_csv(file_path)
    df.columns = ['uid', 'sid', 'rating', 'timestamp']
    return df
