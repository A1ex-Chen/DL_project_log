def area(self) ->torch.Tensor:
    """
        Computes the area of all the boxes.

        Returns:
            torch.Tensor: a vector with areas of each box.
        """
    box = self.tensor
    area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    return area
