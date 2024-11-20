def _parse_labels(self, labels):
    if isinstance(labels, str):
        labels = [label.strip() for label in labels.split(',')]
    return labels
