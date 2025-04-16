def __getitem__(self, item) ->'RotatedBoxes':
    """
        Returns:
            RotatedBoxes: Create a new :class:`RotatedBoxes` by indexing.

        The following usage are allowed:

        1. `new_boxes = boxes[3]`: return a `RotatedBoxes` which contains only one box.
        2. `new_boxes = boxes[2:10]`: return a slice of boxes.
        3. `new_boxes = boxes[vector]`, where vector is a torch.ByteTensor
           with `length = len(boxes)`. Nonzero elements in the vector will be selected.

        Note that the returned RotatedBoxes might share storage with this RotatedBoxes,
        subject to Pytorch's indexing semantics.
        """
    if isinstance(item, int):
        return RotatedBoxes(self.tensor[item].view(1, -1))
    b = self.tensor[item]
    assert b.dim(
        ) == 2, 'Indexing on RotatedBoxes with {} failed to return a matrix!'.format(
        item)
    return RotatedBoxes(b)
