@property
def map(self):
    """
        Mean AP@0.5:0.95 of all classes.

        Return:
            float.
        """
    return self.all_ap.mean() if len(self.all_ap) else 0.0
