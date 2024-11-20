def _get_source_id_from_encoded_image(parsed_tensors):
    return tf.strings.as_string(tf.strings.to_hash_bucket_fast(
        parsed_tensors['image/encoded'], 2 ** 63 - 1))
