def clone(self) ->'RotatedBoxes':
    """
        Clone the RotatedBoxes.

        Returns:
            RotatedBoxes
        """
    return RotatedBoxes(self.tensor.clone())
