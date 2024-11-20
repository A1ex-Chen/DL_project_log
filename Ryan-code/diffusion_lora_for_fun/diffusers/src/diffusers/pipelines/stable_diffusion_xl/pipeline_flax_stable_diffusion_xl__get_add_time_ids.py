def _get_add_time_ids(self, original_size, crops_coords_top_left,
    target_size, bs, dtype):
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = jnp.array([add_time_ids] * bs, dtype=dtype)
    return add_time_ids
