@property
def mp(self):
    """mean precision of all classes.
        Return:
            float.
        """
    return self.p.mean() if len(self.p) else 0.0
