@property
def map75(self):
    """
        Returns the mean Average Precision (mAP) at an IoU threshold of 0.75.

        Returns:
            (float): The mAP at an IoU threshold of 0.75.
        """
    return self.all_ap[:, 5].mean() if len(self.all_ap) else 0.0
