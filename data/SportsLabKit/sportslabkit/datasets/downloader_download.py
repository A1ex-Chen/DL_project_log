def download(self, file_name: (str | None)=None, path: (PathLike | None)=
    _module_path, force: bool=False, quiet: bool=False, unzip: bool=True
    ) ->None:
    """Download the dataset from Kaggle.

        Args:
            file_name (Optional[str], optional): Name of the file to download. If None, downloads all data. Defaults to None.
            path (Optional[PathLike], optional): Path to download the data to. If None, downloads to soccertrack/datasets/data. Defaults to None.
            force (bool, optional): If True, overwrites the existing file. Defaults to False.
            quiet (bool, optional): If True, suppresses the output. Defaults to True.
            unzip (bool, optional): If True, unzips the file. Defaults to True.
        """
    path = Path(path)
    if file_name is None:
        self.api.dataset_download_files(
            f'{self.dataset_owner}/{self.dataset_name}', path=path, force=
            force, quiet=quiet, unzip=unzip)
    else:
        self.api.dataset_download_file(
            f'{self.dataset_owner}/{self.dataset_name}', file_name=
            file_name, path=path, force=force, quiet=quiet)
    if file_name is None and unzip:
        file_name = 'soccertrack'
    if file_name is None and not unzip:
        file_name += 'soccertrack.zip'
    else:
        file_name = Path(file_name).name
    save_path = path / file_name
    return save_path
