@staticmethod
def to_d2_instances_list(instances_list):
    """
        Convert InstancesList to List[Instances]. The input `instances_list` can
        also be a List[Instances], in this case this method is a non-op.
        """
    if not isinstance(instances_list, InstancesList):
        assert all(isinstance(x, Instances) for x in instances_list)
        return instances_list
    ret = []
    for i, info in enumerate(instances_list.im_info):
        instances = Instances(torch.Size([int(info[0].item()), int(info[1].
            item())]))
        ids = instances_list.indices == i
        for k, v in instances_list.batch_extra_fields.items():
            if isinstance(v, torch.Tensor):
                instances.set(k, v[ids])
                continue
            elif isinstance(v, Boxes):
                instances.set(k, v[ids, -4:])
                continue
            target_type, tensor_source = v
            assert isinstance(tensor_source, torch.Tensor)
            assert tensor_source.shape[0] == instances_list.indices.shape[0]
            tensor_source = tensor_source[ids]
            if issubclass(target_type, Boxes):
                instances.set(k, Boxes(tensor_source[:, -4:]))
            elif issubclass(target_type, Keypoints):
                instances.set(k, Keypoints(tensor_source))
            elif issubclass(target_type, torch.Tensor):
                instances.set(k, tensor_source)
            else:
                raise ValueError("Can't handle targe type: {}".format(
                    target_type))
        ret.append(instances)
    return ret
