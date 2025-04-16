def _yield_sim_idx():
    """Generates a dataframe with similarity indices and distances for images."""
    for i in tqdm(range(len(embeddings))):
        sim_idx = self.table.search(embeddings[i]).limit(top_k).to_pandas(
            ).query(f'_distance <= {max_dist}')
        yield [{'idx': i, 'im_file': im_files[i], 'count': len(sim_idx),
            'sim_im_files': sim_idx['im_file'].tolist()}]
