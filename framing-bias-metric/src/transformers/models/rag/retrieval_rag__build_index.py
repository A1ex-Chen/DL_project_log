@staticmethod
def _build_index(config):
    if config.index_name == 'legacy':
        return LegacyIndex(config.retrieval_vector_size, config.index_path or
            LEGACY_INDEX_PATH)
    elif config.index_name == 'custom':
        return CustomHFIndex.load_from_disk(vector_size=config.
            retrieval_vector_size, dataset_path=config.passages_path,
            index_path=config.index_path)
    else:
        return CanonicalHFIndex(vector_size=config.retrieval_vector_size,
            dataset_name=config.dataset, dataset_split=config.dataset_split,
            index_name=config.index_name, index_path=config.index_path,
            use_dummy_dataset=config.use_dummy_dataset)
