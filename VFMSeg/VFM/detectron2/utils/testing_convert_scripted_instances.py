def convert_scripted_instances(instances):
    """
    Convert a scripted Instances object to a regular :class:`Instances` object
    """
    assert hasattr(instances, 'image_size'
        ), f'Expect an Instances object, but got {type(instances)}!'
    ret = Instances(instances.image_size)
    for name in instances._field_names:
        val = getattr(instances, '_' + name, None)
        if val is not None:
            ret.set(name, val)
    return ret
