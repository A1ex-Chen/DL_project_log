@property
def mr(self):
    """mean recall of all classes.
        Return:
            float.
        """
    return self.r.mean() if len(self.r) else 0.0
