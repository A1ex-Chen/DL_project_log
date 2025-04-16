def nested_detach(tensors):
    """Detach `tensors` (even if it's a nested list/tuple of tensors)."""
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    return tensors.detach()
