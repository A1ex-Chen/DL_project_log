def _match_when_rows_are_empty():
    """Performs matching when the rows of similarity matrix are empty.

      When the rows are empty, all detections are false positives. So we return
      a tensor of -1's to indicate that the columns do not match to any rows.

      Returns:
        matches:  int32 tensor indicating the row each column matches to.
      """
    similarity_matrix_shape = shape_utils.combined_static_and_dynamic_shape(
        similarity_matrix)
    return -1 * tf.ones([similarity_matrix_shape[1]], dtype=tf.int32)
