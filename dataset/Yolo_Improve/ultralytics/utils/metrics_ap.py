@property
def ap(self):
    """
        Returns the Average Precision (AP) at an IoU threshold of 0.5-0.95 for all classes.

        Returns:
            (np.ndarray, list): Array of shape (nc,) with AP50-95 values per class, or an empty list if not available.
        """
    return self.all_ap.mean(1) if len(self.all_ap) else []
