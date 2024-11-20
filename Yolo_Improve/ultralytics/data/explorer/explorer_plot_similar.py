def plot_similar(self, img: Union[str, np.ndarray, List[str], List[np.
    ndarray]]=None, idx: Union[int, List[int]]=None, limit: int=25, labels:
    bool=True) ->Image.Image:
    """
        Plot the similar images. Accepts images or indexes.

        Args:
            img (str or list): Path to the image or a list of paths to the images.
            idx (int or list): Index of the image in the table or a list of indexes.
            labels (bool): Whether to plot the labels or not.
            limit (int): Number of results to return. Defaults to 25.

        Returns:
            (PIL.Image): Image containing the plot.

        Example:
            ```python
            exp = Explorer()
            exp.create_embeddings_table()
            similar = exp.plot_similar(img='https://ultralytics.com/images/zidane.jpg')
            ```
        """
    similar = self.get_similar(img, idx, limit, return_type='arrow')
    if len(similar) == 0:
        LOGGER.info('No results found.')
        return None
    img = plot_query_result(similar, plot_labels=labels)
    return Image.fromarray(img)
