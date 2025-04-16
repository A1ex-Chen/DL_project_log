@property
def maps(self):
    """Returns mean Average Precision (mAP) scores per class."""
    return self.box.maps
