def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) ->'Keypoints':
    """
        Create a new `Keypoints` by indexing on this `Keypoints`.

        The following usage are allowed:

        1. `new_kpts = kpts[3]`: return a `Keypoints` which contains only one instance.
        2. `new_kpts = kpts[2:10]`: return a slice of key points.
        3. `new_kpts = kpts[vector]`, where vector is a torch.ByteTensor
           with `length = len(kpts)`. Nonzero elements in the vector will be selected.

        Note that the returned Keypoints might share storage with this Keypoints,
        subject to Pytorch's indexing semantics.
        """
    if isinstance(item, int):
        return Keypoints([self.tensor[item]])
    return Keypoints(self.tensor[item])
