def map_val_table_path(self):
    """
        Map the validation dataset Table like name of file -> it's id in the W&B Table.
        Useful for - referencing artifacts for evaluation.
        """
    self.val_table_path_map = {}
    LOGGER.info('Mapping dataset')
    for i, data in enumerate(tqdm(self.val_table.data)):
        self.val_table_path_map[data[3]] = data[0]
