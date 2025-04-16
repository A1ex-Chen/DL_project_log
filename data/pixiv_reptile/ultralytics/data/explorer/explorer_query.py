def query(self, imgs: Union[str, np.ndarray, List[str], List[np.ndarray]]=
    None, limit: int=25) ->Any:
    """
        Query the table for similar images. Accepts a single image or a list of images.

        Args:
            imgs (str or list): Path to the image or a list of paths to the images.
            limit (int): Number of results to return.

        Returns:
            (pyarrow.Table): An arrow table containing the results. Supports converting to:
                - pandas dataframe: `result.to_pandas()`
                - dict of lists: `result.to_pydict()`

        Example:
            ```python
            exp = Explorer()
            exp.create_embeddings_table()
            similar = exp.query(img='https://ultralytics.com/images/zidane.jpg')
            ```
        """
    if self.table is None:
        raise ValueError('Table is not created. Please create the table first.'
            )
    if isinstance(imgs, str):
        imgs = [imgs]
    assert isinstance(imgs, list
        ), f'img must be a string or a list of strings. Got {type(imgs)}'
    embeds = self.model.embed(imgs)
    embeds = torch.mean(torch.stack(embeds), 0).cpu().numpy() if len(embeds
        ) > 1 else embeds[0].cpu().numpy()
    return self.table.search(embeds).limit(limit).to_arrow()
