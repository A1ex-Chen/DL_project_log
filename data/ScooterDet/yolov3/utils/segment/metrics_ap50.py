@property
def ap50(self):
    """
        AP@0.5 of all classes.

        Return:
            (nc, ) or [].
        """
    return self.all_ap[:, 0] if len(self.all_ap) else []
