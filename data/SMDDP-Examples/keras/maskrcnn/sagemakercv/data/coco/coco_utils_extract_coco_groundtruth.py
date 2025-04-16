def extract_coco_groundtruth(prediction, include_mask=False):
    """Extract COCO format groundtruth.

  Args:
    prediction: dictionary of batch of prediction result. the first dimension
      each element is the batch.
    include_mask: True for including masks in the output annotations.

  Returns:
    Tuple of (images, annotations).
    images: list[dict].Required keys: 'id', 'width' and 'height'. The values are
      image id, width and height.
    annotations: list[dict]. Required keys: {'id', 'source_id', 'category_id',
      'bbox', 'iscrowd'} when include_mask=False. If include_mask=True, also
      required {'area', 'segmentation'}. The 'id' value is the annotation id
      and can be any **positive** number (>=1).
      Refer to http://cocodataset.org/#format-data for more details.
  Raises:
    ValueError: If any groundtruth fields is missing.
  """
    required_fields = ['source_id', 'width', 'height',
        'num_groundtruth_labels', 'groundtruth_boxes', 'groundtruth_classes']
    if include_mask:
        required_fields += ['groundtruth_polygons', 'groundtruth_area']
    for key in required_fields:
        if key not in prediction.keys():
            raise ValueError('Missing groundtruth field: "{}" keys: {}'.
                format(key, prediction.keys()))
    images = []
    annotations = []
    for b in range(prediction['source_id'].shape[0]):
        image = _extract_image_info(prediction, b)
        images.append(image)
        if include_mask:
            flatten_padded_polygons = prediction['groundtruth_polygons'][b]
            flatten_polygons = np.delete(flatten_padded_polygons, np.where(
                flatten_padded_polygons[:] == POLYGON_PAD_VALUE)[0])
            polygons = _unflat_polygons(flatten_polygons)
        num_labels = prediction['num_groundtruth_labels'][b]
        for obj_i in range(num_labels):
            annotation = _extract_bbox_annotation(prediction, b, obj_i)
            if include_mask:
                polygon_info = _extract_polygon_info(prediction, polygons,
                    b, obj_i)
                annotation.update(polygon_info)
            annotations.append(annotation)
    return images, annotations
