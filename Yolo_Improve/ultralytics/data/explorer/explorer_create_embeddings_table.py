def create_embeddings_table(self, force: bool=False, split: str='train'
    ) ->None:
    """
        Create LanceDB table containing the embeddings of the images in the dataset. The table will be reused if it
        already exists. Pass force=True to overwrite the existing table.

        Args:
            force (bool): Whether to overwrite the existing table or not. Defaults to False.
            split (str): Split of the dataset to use. Defaults to 'train'.

        Example:
            ```python
            exp = Explorer()
            exp.create_embeddings_table()
            ```
        """
    if self.table is not None and not force:
        LOGGER.info(
            'Table already exists. Reusing it. Pass force=True to overwrite it.'
            )
        return
    if self.table_name in self.connection.table_names() and not force:
        LOGGER.info(
            f'Table {self.table_name} already exists. Reusing it. Pass force=True to overwrite it.'
            )
        self.table = self.connection.open_table(self.table_name)
        self.progress = 1
        return
    if self.data is None:
        raise ValueError('Data must be provided to create embeddings table')
    data_info = check_det_dataset(self.data)
    if split not in data_info:
        raise ValueError(
            f'Split {split} is not found in the dataset. Available keys in the dataset are {list(data_info.keys())}'
            )
    choice_set = data_info[split]
    choice_set = choice_set if isinstance(choice_set, list) else [choice_set]
    self.choice_set = choice_set
    dataset = ExplorerDataset(img_path=choice_set, data=data_info, augment=
        False, cache=False, task=self.model.task)
    batch = dataset[0]
    vector_size = self.model.embed(batch['im_file'], verbose=False)[0].shape[0]
    table = self.connection.create_table(self.table_name, schema=
        get_table_schema(vector_size), mode='overwrite')
    table.add(self._yield_batches(dataset, data_info, self.model,
        exclude_keys=['img', 'ratio_pad', 'resized_shape', 'ori_shape',
        'batch_idx']))
    self.table = table
