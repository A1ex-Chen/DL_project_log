def plot_similarity_index(self, max_dist: float=0.2, top_k: float=None,
    force: bool=False) ->Image:
    """
        Plot the similarity index of all the images in the table. Here, the index will contain the data points that are
        max_dist or closer to the image in the embedding space at a given index.

        Args:
            max_dist (float): maximum L2 distance between the embeddings to consider. Defaults to 0.2.
            top_k (float): Percentage of closest data points to consider when counting. Used to apply limit when
                running vector search. Defaults to 0.01.
            force (bool): Whether to overwrite the existing similarity index or not. Defaults to True.

        Returns:
            (PIL.Image): Image containing the plot.

        Example:
            ```python
            exp = Explorer()
            exp.create_embeddings_table()

            similarity_idx_plot = exp.plot_similarity_index()
            similarity_idx_plot.show() # view image preview
            similarity_idx_plot.save('path/to/save/similarity_index_plot.png') # save contents to file
            ```
        """
    sim_idx = self.similarity_index(max_dist=max_dist, top_k=top_k, force=force
        )
    sim_count = sim_idx['count'].tolist()
    sim_count = np.array(sim_count)
    indices = np.arange(len(sim_count))
    plt.bar(indices, sim_count)
    plt.xlabel('data idx')
    plt.ylabel('Count')
    plt.title('Similarity Count')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    return Image.fromarray(np.array(Image.open(buffer)))
