@classmethod
def load_from_disk(cls, vector_size, dataset_path, index_path):
    logger.info('Loading passages from {}'.format(dataset_path))
    if dataset_path is None or index_path is None:
        raise ValueError(
            "Please provide ``dataset_path`` and ``index_path`` after calling ``dataset.save_to_disk(dataset_path)`` and ``dataset.get_index('embeddings').save(index_path)``."
            )
    dataset = load_from_disk(dataset_path)
    return cls(vector_size=vector_size, dataset=dataset, index_path=index_path)
