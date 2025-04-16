def create_tf_example(image, bbox_annotations, caption_annotations,
    image_dir, category_index, include_masks=False):
    """Converts image and annotations to a tf.Example proto.

  Args:
    image: dict with keys:
      [u'license', u'file_name', u'coco_url', u'height', u'width',
      u'date_captured', u'flickr_url', u'id']
    bbox_annotations:
      list of dicts with keys:
      [u'segmentation', u'area', u'iscrowd', u'image_id',
      u'bbox', u'category_id', u'id']
      Notice that bounding box coordinates in the official COCO dataset are
      given as [x, y, width, height] tuples using absolute coordinates where
      x, y represent the top-left (0-indexed) corner.  This function converts
      to the format expected by the Tensorflow Object Detection API (which is
      which is [ymin, xmin, ymax, xmax] with coordinates normalized relative
      to image size).
    image_dir: directory containing the image files.
    category_index: a dict containing COCO category information keyed
      by the 'id' field of each category.  See the
      label_map_util.create_category_index function.
    include_masks: Whether to include instance segmentations masks
      (PNG encoded) in the result. default: False.
  Returns:
    example: The converted tf.Example
    num_annotations_skipped: Number of (invalid) annotations that were ignored.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
    image_height = image['height']
    image_width = image['width']
    filename = image['file_name']
    image_id = image['id']
    full_path = os.path.join(image_dir, filename)
    with tf.io.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    key = hashlib.sha256(encoded_jpg).hexdigest()
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    is_crowd = []
    category_names = []
    category_ids = []
    area = []
    encoded_mask_png = []
    num_annotations_skipped = 0
    for object_annotations in bbox_annotations:
        x, y, width, height = tuple(object_annotations['bbox'])
        if width <= 0 or height <= 0:
            num_annotations_skipped += 1
            continue
        if x + width > image_width or y + height > image_height:
            num_annotations_skipped += 1
            continue
        xmin.append(float(x) / image_width)
        xmax.append(float(x + width) / image_width)
        ymin.append(float(y) / image_height)
        ymax.append(float(y + height) / image_height)
        is_crowd.append(object_annotations['iscrowd'])
        category_id = int(object_annotations['category_id'])
        category_ids.append(category_id)
        category_names.append(category_index[category_id]['name'].encode(
            'utf8'))
        area.append(object_annotations['area'])
        if include_masks:
            run_len_encoding = mask.frPyObjects(object_annotations[
                'segmentation'], image_height, image_width)
            binary_mask = mask.decode(run_len_encoding)
            if not object_annotations['iscrowd']:
                binary_mask = np.amax(binary_mask, axis=2)
            pil_image = PIL.Image.fromarray(binary_mask)
            output_io = io.BytesIO()
            pil_image.save(output_io, format='PNG')
            encoded_mask_png.append(output_io.getvalue())
    captions = []
    for caption_annotation in caption_annotations:
        captions.append(caption_annotation['caption'].encode('utf8'))
    feature_dict = {'image/height': dataset_util.int64_feature(image_height
        ), 'image/width': dataset_util.int64_feature(image_width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8'
        )), 'image/source_id': dataset_util.bytes_feature(str(image_id).
        encode('utf8')), 'image/key/sha256': dataset_util.bytes_feature(key
        .encode('utf8')), 'image/encoded': dataset_util.bytes_feature(
        encoded_jpg), 'image/caption': dataset_util.bytes_list_feature(
        captions), 'image/format': dataset_util.bytes_feature('jpeg'.encode
        ('utf8')), 'image/object/bbox/xmin': dataset_util.
        float_list_feature(xmin), 'image/object/bbox/xmax': dataset_util.
        float_list_feature(xmax), 'image/object/bbox/ymin': dataset_util.
        float_list_feature(ymin), 'image/object/bbox/ymax': dataset_util.
        float_list_feature(ymax), 'image/object/class/text': dataset_util.
        bytes_list_feature(category_names), 'image/object/class/label':
        dataset_util.int64_list_feature(category_ids),
        'image/object/is_crowd': dataset_util.int64_list_feature(is_crowd),
        'image/object/area': dataset_util.float_list_feature(area)}
    if include_masks:
        feature_dict['image/object/mask'] = dataset_util.bytes_list_feature(
            encoded_mask_png)
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict)
        )
    return key, example, num_annotations_skipped
