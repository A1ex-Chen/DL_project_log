def _get_coco_panoptic_separated_meta():
    """
    Returns metadata for "separated" version of the panoptic segmentation dataset.
    """
    stuff_ids = [k['id'] for k in COCO_CATEGORIES if k['isthing'] == 0]
    assert len(stuff_ids) == 53, len(stuff_ids)
    stuff_dataset_id_to_contiguous_id = {k: (i + 1) for i, k in enumerate(
        stuff_ids)}
    stuff_dataset_id_to_contiguous_id[0] = 0
    stuff_classes = ['things'] + [k['name'].replace('-other', '').replace(
        '-merged', '') for k in COCO_CATEGORIES if k['isthing'] == 0]
    stuff_colors = [[82, 18, 128]] + [k['color'] for k in COCO_CATEGORIES if
        k['isthing'] == 0]
    ret = {'stuff_dataset_id_to_contiguous_id':
        stuff_dataset_id_to_contiguous_id, 'stuff_classes': stuff_classes,
        'stuff_colors': stuff_colors}
    ret.update(_get_coco_instances_meta())
    return ret
