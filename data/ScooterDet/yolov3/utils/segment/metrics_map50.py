@property
def map50(self):
    """
        Mean AP@0.5 of all classes.

        Return:
            float.
        """
    return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0
