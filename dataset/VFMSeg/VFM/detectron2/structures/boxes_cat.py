@classmethod
def cat(cls, boxes_list: List['Boxes']) ->'Boxes':
    """
        Concatenates a list of Boxes into a single Boxes

        Arguments:
            boxes_list (list[Boxes])

        Returns:
            Boxes: the concatenated Boxes
        """
    assert isinstance(boxes_list, (list, tuple))
    if len(boxes_list) == 0:
        return cls(torch.empty(0))
    assert all([isinstance(box, Boxes) for box in boxes_list])
    cat_boxes = cls(torch.cat([b.tensor for b in boxes_list], dim=0))
    return cat_boxes
