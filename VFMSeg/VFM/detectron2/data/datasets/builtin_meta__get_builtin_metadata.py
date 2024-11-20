def _get_builtin_metadata(dataset_name):
    if dataset_name == 'coco':
        return _get_coco_instances_meta()
    if dataset_name == 'coco_panoptic_separated':
        return _get_coco_panoptic_separated_meta()
    elif dataset_name == 'coco_panoptic_standard':
        meta = {}
        thing_classes = [k['name'] for k in COCO_CATEGORIES]
        thing_colors = [k['color'] for k in COCO_CATEGORIES]
        stuff_classes = [k['name'] for k in COCO_CATEGORIES]
        stuff_colors = [k['color'] for k in COCO_CATEGORIES]
        meta['thing_classes'] = thing_classes
        meta['thing_colors'] = thing_colors
        meta['stuff_classes'] = stuff_classes
        meta['stuff_colors'] = stuff_colors
        thing_dataset_id_to_contiguous_id = {}
        stuff_dataset_id_to_contiguous_id = {}
        for i, cat in enumerate(COCO_CATEGORIES):
            if cat['isthing']:
                thing_dataset_id_to_contiguous_id[cat['id']] = i
            else:
                stuff_dataset_id_to_contiguous_id[cat['id']] = i
        meta['thing_dataset_id_to_contiguous_id'
            ] = thing_dataset_id_to_contiguous_id
        meta['stuff_dataset_id_to_contiguous_id'
            ] = stuff_dataset_id_to_contiguous_id
        return meta
    elif dataset_name == 'coco_person':
        return {'thing_classes': ['person'], 'keypoint_names':
            COCO_PERSON_KEYPOINT_NAMES, 'keypoint_flip_map':
            COCO_PERSON_KEYPOINT_FLIP_MAP, 'keypoint_connection_rules':
            KEYPOINT_CONNECTION_RULES}
    elif dataset_name == 'cityscapes':
        CITYSCAPES_THING_CLASSES = ['person', 'rider', 'car', 'truck',
            'bus', 'train', 'motorcycle', 'bicycle']
        CITYSCAPES_STUFF_CLASSES = ['road', 'sidewalk', 'building', 'wall',
            'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation',
            'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus',
            'train', 'motorcycle', 'bicycle']
        return {'thing_classes': CITYSCAPES_THING_CLASSES, 'stuff_classes':
            CITYSCAPES_STUFF_CLASSES}
    raise KeyError('No built-in metadata for dataset {}'.format(dataset_name))
