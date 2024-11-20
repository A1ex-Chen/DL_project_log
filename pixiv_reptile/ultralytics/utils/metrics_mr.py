@property
def mr(self):
    """
        Returns the Mean Recall of all classes.

        Returns:
            (float): The mean recall of all classes.
        """
    return self.r.mean() if len(self.r) else 0.0
