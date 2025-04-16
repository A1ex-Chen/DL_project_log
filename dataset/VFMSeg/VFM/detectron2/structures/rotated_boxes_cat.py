@classmethod
def cat(cls, boxes_list: List['RotatedBoxes']) ->'RotatedBoxes':
    """
        Concatenates a list of RotatedBoxes into a single RotatedBoxes

        Arguments:
            boxes_list (list[RotatedBoxes])

        Returns:
            RotatedBoxes: the concatenated RotatedBoxes
        """
    assert isinstance(boxes_list, (list, tuple))
    if len(boxes_list) == 0:
        return cls(torch.empty(0))
    assert all([isinstance(box, RotatedBoxes) for box in boxes_list])
    cat_boxes = cls(torch.cat([b.tensor for b in boxes_list], dim=0))
    return cat_boxes
