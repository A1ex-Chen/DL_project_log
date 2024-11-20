def plot_training_labels(self):
    """Create a labeled training plot of the YOLO model."""
    boxes = np.concatenate([lb['bboxes'] for lb in self.train_loader.
        dataset.labels], 0)
    cls = np.concatenate([lb['cls'] for lb in self.train_loader.dataset.
        labels], 0)
    plot_labels(boxes, cls.squeeze(), names=self.data['names'], save_dir=
        self.save_dir, on_plot=self.on_plot)
