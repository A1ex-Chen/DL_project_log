def process_global_cap(self, img_id):
    if self._captions_reader is not None:
        obj_count = {}
        for _, cap in self.cap_cache[img_id].items():
            for c in cap['used_cls']:
                if c not in obj_count:
                    obj_count[c] = 0
                obj_count[c] += 1
        sort_obj = sorted(obj_count.items(), key=lambda x: x[1], reverse=True)
        top_obj = [x[0] for x in sort_obj[:2]]
    else:
        top_obj = []
    mention_flag = np.zeros((1, self.obj_cache[img_id]['encoder_input_ids']
        .shape[0]), dtype=np.int64)
    if self.cbs_class is not None:
        top_obj = self.cbs_class[img_id]
    for index, ecls in enumerate(self.obj_cache[img_id]['encoder_cls'].tolist()
        ):
        if ecls in top_obj:
            mention_flag[0, index] = 1
        elif ecls < 1601:
            mention_flag[0, index] = 3
    self.global_obj_cache[img_id] = mention_flag
