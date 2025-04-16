def _decode_masks(self, parsed_tensors):
    """Decode a set of PNG masks to the tf.float32 tensors."""

    def _decode_png_mask(png_bytes):
        mask = tf.squeeze(tf.io.decode_png(png_bytes, channels=1, dtype=tf.
            uint8), axis=-1)
        mask = tf.cast(mask, dtype=tf.float32)
        mask.set_shape([None, None])
        return mask
    height = parsed_tensors['image/height']
    width = parsed_tensors['image/width']
    masks = parsed_tensors['image/object/mask']
    return tf.cond(tf.greater(tf.size(masks), 0), lambda : tf.map_fn(
        _decode_png_mask, masks, dtype=tf.float32), lambda : tf.zeros([0,
        height, width], dtype=tf.float32))
