def __getitem__(self, item) ->'Boxes':
    """
        Args:
            item: int, slice, or a BoolTensor

        Returns:
            Boxes: Create a new :class:`Boxes` by indexing.

        The following usage are allowed:

        1. `new_boxes = boxes[3]`: return a `Boxes` which contains only one box.
        2. `new_boxes = boxes[2:10]`: return a slice of boxes.
        3. `new_boxes = boxes[vector]`, where vector is a torch.BoolTensor
           with `length = len(boxes)`. Nonzero elements in the vector will be selected.

        Note that the returned Boxes might share storage with this Boxes,
        subject to Pytorch's indexing semantics.
        """
    if isinstance(item, int):
        return Boxes(self.tensor[item].view(1, -1))
    b = self.tensor[item]
    assert b.dim(
        ) == 2, 'Indexing on Boxes with {} failed to return a matrix!'.format(
        item)
    return Boxes(b)
