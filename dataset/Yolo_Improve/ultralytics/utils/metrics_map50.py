@property
def map50(self):
    """
        Returns the mean Average Precision (mAP) at an IoU threshold of 0.5.

        Returns:
            (float): The mAP at an IoU threshold of 0.5.
        """
    return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0
