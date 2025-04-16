def collate(self, instances):
    """collate function for dataloader"""
    vision_types = [instance['vision_type'] for instance in instances]
    instances_2d, instances_3d = [], []
    for i, vision_type in enumerate(vision_types):
        assert vision_type in self.vision_type
        if vision_type == 'image':
            instances_2d.append(instances[i])
        else:
            instances_3d.append(instances[i])
    return_dict = {}
    if len(instances_2d) > 0:
        instances_2d = self.collate_2d(instances_2d)
        return_dict['image'] = instances_2d
    else:
        return_dict['image'] = None
    if len(instances_3d) > 0:
        instances_3d = self.collate_3d(instances_3d)
        return_dict['pcl'] = instances_3d
    else:
        return_dict['pcl'] = None
    return return_dict
