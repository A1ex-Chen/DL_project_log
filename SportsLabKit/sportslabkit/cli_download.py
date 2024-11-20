def download(self, dataset: str='all', output: str='./data', quiet: bool=False
    ):
    """Download data from google drive

        Args:
            dataset (str, optional): Dataset to download. Defaults to "all".
            output (str, optional): Where to save the data. Defaults to "./data".
            quiet (bool, optional): Whether to silence the output. Defaults to False.

        Raises:
            ValueError: _description_
        """
    if dataset == 'all':
        url = (
            'https://drive.google.com/drive/u/1/folders/13bk0oSsH0WL9LBmr9_4zYn6WqfntT3qF'
            )
    else:
        raise ValueError('Dataset not found.')
    gdown.download_folder(url=url, output=output, quiet=quiet, use_cookies=
        False)
