@staticmethod
def cat(instance_lists: List['Instances']) ->'Instances':
    """
        Args:
            instance_lists (list[Instances])

        Returns:
            Instances
        """
    assert all(isinstance(i, Instances) for i in instance_lists)
    assert len(instance_lists) > 0
    if len(instance_lists) == 1:
        return instance_lists[0]
    image_size = instance_lists[0].image_size
    if not isinstance(image_size, torch.Tensor):
        for i in instance_lists[1:]:
            assert i.image_size == image_size
    ret = Instances(image_size)
    for k in instance_lists[0]._fields.keys():
        values = [i.get(k) for i in instance_lists]
        v0 = values[0]
        if isinstance(v0, torch.Tensor):
            values = torch.cat(values, dim=0)
        elif isinstance(v0, list):
            values = list(itertools.chain(*values))
        elif hasattr(type(v0), 'cat'):
            values = type(v0).cat(values)
        else:
            raise ValueError('Unsupported type {} for concatenation'.format
                (type(v0)))
        ret.set(k, values)
    return ret
