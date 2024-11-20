def preprocess(self):
    dataset_path = self._get_preprocessed_dataset_path()
    if dataset_path.is_file():
        print('Already preprocessed. Skip preprocessing')
        return
    if not dataset_path.parent.is_dir():
        dataset_path.parent.mkdir(parents=True)
    self.maybe_download_raw_dataset()
    df = self.load_ratings_df()
    df = self.make_implicit(df)
    df = self.filter_triplets(df)
    df, umap, smap = self.densify_index(df)
    train, val, test = self.split_df(df, len(umap))
    dataset = {'train': train, 'val': val, 'test': test, 'umap': umap,
        'smap': smap}
    with dataset_path.open('wb') as f:
        pickle.dump(dataset, f)
