def _is_dead_end(versioned_blob):
    return not (versioned_blob in versioned_external_output or len(
        consumer_map[versioned_blob]) > 0 and all(x[0] not in
        removed_op_ids for x in consumer_map[versioned_blob]))
