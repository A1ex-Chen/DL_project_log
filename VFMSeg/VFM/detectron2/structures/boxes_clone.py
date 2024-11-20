def clone(self) ->'Boxes':
    """
        Clone the Boxes.

        Returns:
            Boxes
        """
    return Boxes(self.tensor.clone())
