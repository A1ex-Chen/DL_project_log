def create_keypoint_hflip_indices(dataset_names: Union[str, List[str]]) ->List[
    int]:
    """
    Args:
        dataset_names: list of dataset names

    Returns:
        list[int]: a list of size=#keypoints, storing the
        horizontally-flipped keypoint indices.
    """
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    check_metadata_consistency('keypoint_names', dataset_names)
    check_metadata_consistency('keypoint_flip_map', dataset_names)
    meta = MetadataCatalog.get(dataset_names[0])
    names = meta.keypoint_names
    flip_map = dict(meta.keypoint_flip_map)
    flip_map.update({v: k for k, v in flip_map.items()})
    flipped_names = [(i if i not in flip_map else flip_map[i]) for i in names]
    flip_indices = [names.index(i) for i in flipped_names]
    return flip_indices
