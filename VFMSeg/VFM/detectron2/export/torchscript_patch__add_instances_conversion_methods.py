def _add_instances_conversion_methods(newInstances):
    """
    Add from_instances methods to the scripted Instances class.
    """
    cls_name = newInstances.__name__

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
    newInstances.from_instances = from_instances
