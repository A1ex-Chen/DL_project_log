@property
def ap(self):
    """AP@0.5:0.95
        Return:
            (nc, ) or [].
        """
    return self.all_ap.mean(1) if len(self.all_ap) else []
