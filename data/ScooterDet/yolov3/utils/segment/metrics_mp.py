@property
def mp(self):
    """
        Mean precision of all classes.

        Return:
            float.
        """
    return self.p.mean() if len(self.p) else 0.0
