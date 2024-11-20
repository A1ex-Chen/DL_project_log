def __getitem__(self, index):
    if self.is_training:
        img_id, cap, gt = self._captions_reader[index]
    else:
        img_id = self._image_ids[index]
        if self._captions_reader is not None:
            gt = self._captions_reader.get_gt_by_image_id(img_id)
        else:
            gt = None
        cap = None
    if img_id not in self.obj_cache:
        self.process_obj(img_id)
    item = self.obj_cache[img_id]
    if gt is not None:
        item['gt'] = gt
    if cap is not None:
        if cap not in self.cap_cache:
            self.process_cap(img_id, cap)
        item['cap'] = self.cap_cache[img_id][cap]['input_ids']
        item['mention_flag'] = self.cap_cache[img_id][cap]['mention_flag']
        item['mention_flag'][:, -1] = 0
    else:
        item['mention_flag'] = self.global_obj_cache[img_id]
        item['mention_flag'][:, -1] = 0
    return item
