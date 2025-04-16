def _match_when_rows_are_non_empty():
    """Performs matching when the rows of similarity matrix are non empty.

      Returns:
        matches:  int32 tensor indicating the row each column matches to.
      """
    matches = tf.argmax(input=similarity_matrix, axis=0, output_type=tf.int32)
    if self._matched_threshold is not None:
        matched_vals = tf.reduce_max(input_tensor=similarity_matrix, axis=0)
        below_unmatched_threshold = tf.greater(self._unmatched_threshold,
            matched_vals)
        between_thresholds = tf.logical_and(tf.greater_equal(matched_vals,
            self._unmatched_threshold), tf.greater(self._matched_threshold,
            matched_vals))
        if self._negatives_lower_than_unmatched:
            matches = self._set_values_using_indicator(matches,
                below_unmatched_threshold, -1)
            matches = self._set_values_using_indicator(matches,
                between_thresholds, -2)
        else:
            matches = self._set_values_using_indicator(matches,
                below_unmatched_threshold, -2)
            matches = self._set_values_using_indicator(matches,
                between_thresholds, -1)
    if self._force_match_for_each_row:
        similarity_matrix_shape = (shape_utils.
            combined_static_and_dynamic_shape(similarity_matrix))
        force_match_column_ids = tf.argmax(input=similarity_matrix, axis=1,
            output_type=tf.int32)
        force_match_column_indicators = tf.one_hot(force_match_column_ids,
            depth=similarity_matrix_shape[1])
        force_match_row_ids = tf.argmax(input=force_match_column_indicators,
            axis=0, output_type=tf.int32)
        force_match_column_mask = tf.cast(tf.reduce_max(input_tensor=
            force_match_column_indicators, axis=0), tf.bool)
        final_matches = tf.where(force_match_column_mask,
            force_match_row_ids, matches)
        return final_matches
    else:
        return matches
