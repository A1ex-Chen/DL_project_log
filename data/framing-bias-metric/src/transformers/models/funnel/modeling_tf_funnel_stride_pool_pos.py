def stride_pool_pos(self, pos_id, block_index):
    """
        Pool `pos_id` while keeping the cls token separate (if `self.separate_cls=True`).
        """
    if self.separate_cls:
        cls_pos = tf.constant([-2 ** block_index + 1], dtype=pos_id.dtype)
        pooled_pos_id = pos_id[1:-1] if self.truncate_seq else pos_id[1:]
        return tf.concat([cls_pos, pooled_pos_id[::2]], 0)
    else:
        return pos_id[::2]
