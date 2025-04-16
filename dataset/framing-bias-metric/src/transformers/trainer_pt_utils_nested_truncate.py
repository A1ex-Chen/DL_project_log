def nested_truncate(tensors, limit):
    """Truncate `tensors` at `limit` (even if it's a nested list/tuple of tensors)."""
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_truncate(t, limit) for t in tensors)
    return tensors[:limit]
