def _apply_moving(v_moving, v_normal):
    diff = v_moving - v_normal
    v_moving.assign_sub(tf.cast(1.0 - decay, v_moving.dtype) * diff)
    return v_moving
