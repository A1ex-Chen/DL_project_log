@torch.jit.unused
def from_instances(instances: Instances):
    """
        Create scripted Instances from original Instances
        """
    fields = instances.get_fields()
    image_size = instances.image_size
    ret = newInstances(image_size)
    for name, val in fields.items():
        assert hasattr(ret, f'_{name}'
            ), f'No attribute named {name} in {cls_name}'
        setattr(ret, name, deepcopy(val))
    return ret
