@property
def conf(self):
    """Gets the confidence values of Oriented Bounding Boxes (OBBs)."""
    return self.data[:, -2]
