def convert_to_coco_dict(dataset_name):
    """
    Convert an instance detection/segmentation or keypoint detection dataset
    in detectron2's standard format into COCO json format.

    Generic dataset description can be found here:
    https://detectron2.readthedocs.io/tutorials/datasets.html#register-a-dataset

    COCO data format description can be found here:
    http://cocodataset.org/#format-data

    Args:
        dataset_name (str):
            name of the source dataset
            Must be registered in DatastCatalog and in detectron2's standard format.
            Must have corresponding metadata "thing_classes"
    Returns:
        coco_dict: serializable dict in COCO json format
    """
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    if hasattr(metadata, 'thing_dataset_id_to_contiguous_id'):
        reverse_id_mapping = {v: k for k, v in metadata.
            thing_dataset_id_to_contiguous_id.items()}
        reverse_id_mapper = lambda contiguous_id: reverse_id_mapping[
            contiguous_id]
    else:
        reverse_id_mapper = lambda contiguous_id: contiguous_id
    categories = [{'id': reverse_id_mapper(id), 'name': name} for id, name in
        enumerate(metadata.thing_classes)]
    logger.info('Converting dataset dicts into COCO format')
    coco_images = []
    coco_annotations = []
    for image_id, image_dict in enumerate(dataset_dicts):
        coco_image = {'id': image_dict.get('image_id', image_id), 'width':
            int(image_dict['width']), 'height': int(image_dict['height']),
            'file_name': str(image_dict['file_name'])}
        coco_images.append(coco_image)
        anns_per_image = image_dict.get('annotations', [])
        for annotation in anns_per_image:
            coco_annotation = {}
            bbox = annotation['bbox']
            if isinstance(bbox, np.ndarray):
                if bbox.ndim != 1:
                    raise ValueError(
                        f'bbox has to be 1-dimensional. Got shape={bbox.shape}.'
                        )
                bbox = bbox.tolist()
            if len(bbox) not in [4, 5]:
                raise ValueError(f'bbox has to has length 4 or 5. Got {bbox}.')
            from_bbox_mode = annotation['bbox_mode']
            to_bbox_mode = BoxMode.XYWH_ABS if len(bbox
                ) == 4 else BoxMode.XYWHA_ABS
            bbox = BoxMode.convert(bbox, from_bbox_mode, to_bbox_mode)
            if 'segmentation' in annotation:
                segmentation = annotation['segmentation']
                if isinstance(segmentation, list):
                    polygons = PolygonMasks([segmentation])
                    area = polygons.area()[0].item()
                elif isinstance(segmentation, dict):
                    area = mask_util.area(segmentation).item()
                else:
                    raise TypeError(
                        f'Unknown segmentation type {type(segmentation)}!')
            elif to_bbox_mode == BoxMode.XYWH_ABS:
                bbox_xy = BoxMode.convert(bbox, to_bbox_mode, BoxMode.XYXY_ABS)
                area = Boxes([bbox_xy]).area()[0].item()
            else:
                area = RotatedBoxes([bbox]).area()[0].item()
            if 'keypoints' in annotation:
                keypoints = annotation['keypoints']
                for idx, v in enumerate(keypoints):
                    if idx % 3 != 2:
                        keypoints[idx] = v - 0.5
                if 'num_keypoints' in annotation:
                    num_keypoints = annotation['num_keypoints']
                else:
                    num_keypoints = sum(kp > 0 for kp in keypoints[2::3])
            coco_annotation['id'] = len(coco_annotations) + 1
            coco_annotation['image_id'] = coco_image['id']
            coco_annotation['bbox'] = [round(float(x), 3) for x in bbox]
            coco_annotation['area'] = float(area)
            coco_annotation['iscrowd'] = int(annotation.get('iscrowd', 0))
            coco_annotation['category_id'] = int(reverse_id_mapper(
                annotation['category_id']))
            if 'keypoints' in annotation:
                coco_annotation['keypoints'] = keypoints
                coco_annotation['num_keypoints'] = num_keypoints
            if 'segmentation' in annotation:
                seg = coco_annotation['segmentation'] = annotation[
                    'segmentation']
                if isinstance(seg, dict):
                    counts = seg['counts']
                    if not isinstance(counts, str):
                        seg['counts'] = counts.decode('ascii')
            coco_annotations.append(coco_annotation)
    logger.info(
        f'Conversion finished, #images: {len(coco_images)}, #annotations: {len(coco_annotations)}'
        )
    info = {'date_created': str(datetime.datetime.now()), 'description':
        'Automatically generated COCO json file for Detectron2.'}
    coco_dict = {'info': info, 'images': coco_images, 'categories':
        categories, 'licenses': None}
    if len(coco_annotations) > 0:
        coco_dict['annotations'] = coco_annotations
    return coco_dict
