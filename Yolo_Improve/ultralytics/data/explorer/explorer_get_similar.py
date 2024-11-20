def get_similar(self, img: Union[str, np.ndarray, List[str], List[np.
    ndarray]]=None, idx: Union[int, List[int]]=None, limit: int=25,
    return_type: str='pandas') ->Any:
    """
        Query the table for similar images. Accepts a single image or a list of images.

        Args:
            img (str or list): Path to the image or a list of paths to the images.
            idx (int or list): Index of the image in the table or a list of indexes.
            limit (int): Number of results to return. Defaults to 25.
            return_type (str): Type of the result to return. Can be either 'pandas' or 'arrow'. Defaults to 'pandas'.

        Returns:
            (pandas.DataFrame): A dataframe containing the results.

        Example:
            ```python
            exp = Explorer()
            exp.create_embeddings_table()
            similar = exp.get_similar(img='https://ultralytics.com/images/zidane.jpg')
            ```
        """
    assert return_type in {'pandas', 'arrow'
        }, f'Return type should be `pandas` or `arrow`, but got {return_type}'
    img = self._check_imgs_or_idxs(img, idx)
    similar = self.query(img, limit=limit)
    if return_type == 'arrow':
        return similar
    elif return_type == 'pandas':
        return similar.to_pandas()
