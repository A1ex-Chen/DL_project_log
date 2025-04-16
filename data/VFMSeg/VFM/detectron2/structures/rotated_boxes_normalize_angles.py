def normalize_angles(self) ->None:
    """
        Restrict angles to the range of [-180, 180) degrees
        """
    self.tensor[:, 4] = (self.tensor[:, 4] + 180.0) % 360.0 - 180.0
