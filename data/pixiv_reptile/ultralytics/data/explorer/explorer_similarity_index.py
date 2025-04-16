def similarity_index(self, max_dist: float=0.2, top_k: float=None, force:
    bool=False) ->Any:
    """
        Calculate the similarity index of all the images in the table. Here, the index will contain the data points that
        are max_dist or closer to the image in the embedding space at a given index.

        Args:
            max_dist (float): maximum L2 distance between the embeddings to consider. Defaults to 0.2.
            top_k (float): Percentage of the closest data points to consider when counting. Used to apply limit.
                           vector search. Defaults: None.
            force (bool): Whether to overwrite the existing similarity index or not. Defaults to True.

        Returns:
            (pandas.DataFrame): A dataframe containing the similarity index. Each row corresponds to an image,
                and columns include indices of similar images and their respective distances.

        Example:
            ```python
            exp = Explorer()
            exp.create_embeddings_table()
            sim_idx = exp.similarity_index()
            ```
        """
    if self.table is None:
        raise ValueError('Table is not created. Please create the table first.'
            )
    sim_idx_table_name = (
        f'{self.sim_idx_base_name}_thres_{max_dist}_top_{top_k}'.lower())
    if sim_idx_table_name in self.connection.table_names() and not force:
        LOGGER.info(
            'Similarity matrix already exists. Reusing it. Pass force=True to overwrite it.'
            )
        return self.connection.open_table(sim_idx_table_name).to_pandas()
    if top_k and not 1.0 >= top_k >= 0.0:
        raise ValueError(f'top_k must be between 0.0 and 1.0. Got {top_k}')
    if max_dist < 0.0:
        raise ValueError(f'max_dist must be greater than 0. Got {max_dist}')
    top_k = int(top_k * len(self.table)) if top_k else len(self.table)
    top_k = max(top_k, 1)
    features = self.table.to_lance().to_table(columns=['vector', 'im_file']
        ).to_pydict()
    im_files = features['im_file']
    embeddings = features['vector']
    sim_table = self.connection.create_table(sim_idx_table_name, schema=
        get_sim_index_schema(), mode='overwrite')

    def _yield_sim_idx():
        """Generates a dataframe with similarity indices and distances for images."""
        for i in tqdm(range(len(embeddings))):
            sim_idx = self.table.search(embeddings[i]).limit(top_k).to_pandas(
                ).query(f'_distance <= {max_dist}')
            yield [{'idx': i, 'im_file': im_files[i], 'count': len(sim_idx),
                'sim_im_files': sim_idx['im_file'].tolist()}]
    sim_table.add(_yield_sim_idx())
    self.sim_index = sim_table
    return sim_table.to_pandas()
