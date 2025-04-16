def draw_sem_seg(self, sem_seg, area_threshold=None, alpha=0.8):
    """
        Draw semantic segmentation predictions/labels.

        Args:
            sem_seg (Tensor or ndarray): the segmentation of shape (H, W).
                Each value is the integer label of the pixel.
            area_threshold (int): segments with less than `area_threshold` are not drawn.
            alpha (float): the larger it is, the more opaque the segmentations are.

        Returns:
            output (VisImage): image object with visualizations.
        """
    if isinstance(sem_seg, torch.Tensor):
        sem_seg = sem_seg.numpy()
    labels, areas = np.unique(sem_seg, return_counts=True)
    sorted_idxs = np.argsort(-areas).tolist()
    labels = labels[sorted_idxs]
    for label in filter(lambda l: l < len(self.metadata.stuff_classes), labels
        ):
        try:
            mask_color = [(x / 255) for x in self.metadata.stuff_colors[label]]
        except (AttributeError, IndexError):
            mask_color = None
        binary_mask = (sem_seg == label).astype(np.uint8)
        text = self.metadata.stuff_classes[label]
        self.draw_binary_mask(binary_mask, color=mask_color, edge_color=
            _OFF_WHITE, text=text, alpha=alpha, area_threshold=area_threshold)
    return self.output
