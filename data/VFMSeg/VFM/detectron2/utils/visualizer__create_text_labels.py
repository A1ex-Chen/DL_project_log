def _create_text_labels(classes, scores, class_names, is_crowd=None):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):
        is_crowd (list[bool] or None):

    Returns:
        list[str] or None
    """
    labels = None
    if classes is not None:
        if class_names is not None and len(class_names) > 0:
            labels = [class_names[i] for i in classes]
        else:
            labels = [str(i) for i in classes]
    if scores is not None:
        if labels is None:
            labels = ['{:.0f}%'.format(s * 100) for s in scores]
        else:
            labels = ['{} {:.0f}%'.format(l, s * 100) for l, s in zip(
                labels, scores)]
    if labels is not None and is_crowd is not None:
        labels = [(l + ('|crowd' if crowd else '')) for l, crowd in zip(
            labels, is_crowd)]
    return labels
