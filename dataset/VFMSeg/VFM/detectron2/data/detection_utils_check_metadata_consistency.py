def check_metadata_consistency(key, dataset_names):
    """
    Check that the datasets have consistent metadata.

    Args:
        key (str): a metadata key
        dataset_names (list[str]): a list of dataset names

    Raises:
        AttributeError: if the key does not exist in the metadata
        ValueError: if the given datasets do not have the same metadata values defined by key
    """
    if len(dataset_names) == 0:
        return
    logger = logging.getLogger(__name__)
    entries_per_dataset = [getattr(MetadataCatalog.get(d), key) for d in
        dataset_names]
    for idx, entry in enumerate(entries_per_dataset):
        if entry != entries_per_dataset[0]:
            logger.error("Metadata '{}' for dataset '{}' is '{}'".format(
                key, dataset_names[idx], str(entry)))
            logger.error("Metadata '{}' for dataset '{}' is '{}'".format(
                key, dataset_names[0], str(entries_per_dataset[0])))
            raise ValueError("Datasets have different metadata '{}'!".
                format(key))
