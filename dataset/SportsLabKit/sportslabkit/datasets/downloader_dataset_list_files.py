def dataset_list_files(self) ->None:
    """List the files in the dataset."""
    files = self.api.dataset_list_files(
        f'{self.dataset_owner}/{self.dataset_name}')
    logger.info(files)
