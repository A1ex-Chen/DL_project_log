def _extract_polygon_info(prediction, polygons, b, obj_i):
    """Constructs 'area' and 'segmentation' fields.

  Args:
    prediction: dict[str, numpy.array]. Model outputs. The value dimension is
      [batch_size, #objects, #features, ...]
    polygons: list[list[list]]. Dimensions are [#objects, #polygon, #vertex].
    b: batch index.
    obj_i: object index.

  Returns:
    dict[str, numpy.array]. COCO format annotation with 'area' and
    'segmentation'.
  """
    annotation = {}
    if 'groundtruth_area' in prediction:
        groundtruth_area = float(prediction['groundtruth_area'][b][obj_i])
    else:
        height = prediction['height'][b]
        width = prediction['width'][b]
        rles = coco_mask.frPyObjects(polygons[obj_i], height, width)
        groundtruth_area = coco_mask.area(rles)
    annotation['area'] = groundtruth_area
    annotation['segmentation'] = polygons[obj_i]
    if not annotation['segmentation'][0]:
        height = prediction['height'][b]
        width = prediction['width'][b]
        bbox = _denormalize_to_coco_bbox(prediction['groundtruth_boxes'][b]
            [obj_i, :], height, width)
        xcenter = bbox[0] + bbox[2] / 2.0
        ycenter = bbox[1] + bbox[3] / 2.0
        annotation['segmentation'] = [[xcenter, ycenter, xcenter, ycenter,
            xcenter, ycenter, xcenter, ycenter]]
    return annotation
