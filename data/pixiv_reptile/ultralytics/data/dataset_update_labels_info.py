def update_labels_info(self, label):
    """Add texts information for multi-modal model training."""
    labels = super().update_labels_info(label)
    labels['texts'] = [v.split('/') for _, v in self.data['names'].items()]
    return labels
