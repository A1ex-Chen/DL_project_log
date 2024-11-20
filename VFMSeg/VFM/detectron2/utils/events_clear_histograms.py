def clear_histograms(self):
    """
        Delete all the stored histograms for visualization.
        This should be called after histograms are written to tensorboard.
        """
    self._histograms = []
