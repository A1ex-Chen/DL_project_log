@property
def mp(self):
    """
        Returns the Mean Precision of all classes.

        Returns:
            (float): The mean precision of all classes.
        """
    return self.p.mean() if len(self.p) else 0.0
