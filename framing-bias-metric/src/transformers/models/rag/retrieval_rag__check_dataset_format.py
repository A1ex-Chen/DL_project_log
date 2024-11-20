def _check_dataset_format(self, with_index: bool):
    if not isinstance(self.dataset, Dataset):
        raise ValueError(
            'Dataset should be a datasets.Dataset object, but got {}'.
            format(type(self.dataset)))
    if len({'title', 'text', 'embeddings'} - set(self.dataset.column_names)
        ) > 0:
        raise ValueError(
            'Dataset should be a dataset with the following columns: title (str), text (str) and embeddings (arrays of dimension vector_size), but got columns {}'
            .format(self.dataset.column_names))
    if with_index and 'embeddings' not in self.dataset.list_indexes():
        raise ValueError(
            'Missing faiss index in the dataset. Make sure you called `dataset.add_faiss_index` to compute it or `dataset.load_faiss_index` to load one from the disk.'
            )
