@property
def ap50(self):
    """
        Returns the Average Precision (AP) at an IoU threshold of 0.5 for all classes.

        Returns:
            (np.ndarray, list): Array of shape (nc,) with AP50 values per class, or an empty list if not available.
        """
    return self.all_ap[:, 0] if len(self.all_ap) else []
