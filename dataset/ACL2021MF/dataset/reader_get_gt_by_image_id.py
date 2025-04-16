def get_gt_by_image_id(self, image_id):
    caps = self._captions_dict[image_id]
    return [c['gt'] for c in caps]
