def tf_calculate_resize_scale(in_size, out_size):
    if align_corners:
        if is_tracing:
            return (in_size - 1) / (out_size.float() - 1).clamp(min=1)
        else:
            return (in_size - 1) / max(1, out_size - 1)
    elif is_tracing:
        return in_size / out_size.float()
    else:
        return in_size / out_size
