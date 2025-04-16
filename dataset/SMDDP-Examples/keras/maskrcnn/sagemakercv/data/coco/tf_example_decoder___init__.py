def __init__(self, use_instance_mask=False, regenerate_source_id=False):
    self._include_mask = use_instance_mask
    self._regenerate_source_id = regenerate_source_id
    self._keys_to_features = {'image/encoded': tf.io.FixedLenFeature((), tf
        .string), 'image/source_id': tf.io.FixedLenFeature((), tf.string),
        'image/height': tf.io.FixedLenFeature((), tf.int64), 'image/width':
        tf.io.FixedLenFeature((), tf.int64), 'image/object/bbox/xmin': tf.
        io.VarLenFeature(tf.float32), 'image/object/bbox/xmax': tf.io.
        VarLenFeature(tf.float32), 'image/object/bbox/ymin': tf.io.
        VarLenFeature(tf.float32), 'image/object/bbox/ymax': tf.io.
        VarLenFeature(tf.float32), 'image/object/class/label': tf.io.
        VarLenFeature(tf.int64), 'image/object/area': tf.io.VarLenFeature(
        tf.float32), 'image/object/is_crowd': tf.io.VarLenFeature(tf.int64)}
    if use_instance_mask:
        self._keys_to_features.update({'image/object/mask': tf.io.
            VarLenFeature(tf.string)})
