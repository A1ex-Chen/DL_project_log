def collate_3d(self, instances):
    keys = [key for key in instances[0].keys() if key != 'vision_type']
    return_dict = defaultdict()
    return_dict['vision_type'] = 'pcl'
    for key in keys:
        return_dict[key] = []
        for instance in instances:
            return_dict[key].append(instance[key])
        if isinstance(instance[key], torch.Tensor):
            if key == 'scene_scale':
                return_dict[key] = torch.stack(return_dict[key])
            else:
                return_dict[key] = torch.stack(return_dict[key]).half()
    return return_dict
