def process_input_for_encoder(self, img_id, obj_features, obj_boxes,
    obj_cls, copy_obj_features=None, copy_obj_boxes=None, copy_obj_cls=None):
    obj_size = obj_features.shape[0]
    cls2objindex = {}
    for obj_index, cls_ in enumerate(obj_cls):
        if (self._blacklist_categories is not None and self.copy_vocab.
            id_to_category[cls_].lower() in self._blacklist_categories):
            cls_ = 0
        if cls_ not in cls2objindex:
            cls2objindex[cls_] = []
        cls2objindex[cls_].append((obj_index, obj_boxes[obj_index, 6]))
    if self.config.use_copy_obj:
        for obj_index, cls_ in enumerate(copy_obj_cls):
            if (self._blacklist_categories is not None and self.copy_vocab.
                id_to_category[cls_].lower() in self._blacklist_categories):
                cls_ = 0
            if cls_ not in cls2objindex:
                cls2objindex[cls_] = []
            cls2objindex[cls_].append((obj_index + obj_size, copy_obj_boxes
                [obj_index, 6]))
    for cls_ in cls2objindex:
        cls2objindex[cls_] = sorted(cls2objindex[cls_], key=lambda x: x[1])
    encoder_input_ids = []
    encoder_img_mask = []
    encoder_cls = []
    rel_position = []
    img_order = []
    key_order = sorted([k for k in cls2objindex.keys()])
    for cls_ in key_order:
        rel_position_list = []
        if cls_ == 0:
            input_ids = [self.background_index]
        else:
            input_ids = self.copy_vocab.token_class[cls_]
        for img_i in range(len(cls2objindex[cls_])):
            each_img_rel = [48] * len(cls2objindex[cls_])
            each_img_rel[img_i] = 0
            each_img_rel += [(31 + get_position_emb_index(w_i + 1)) for w_i in
                range(len(input_ids))]
            rel_position_list.append(each_img_rel)
        for word_i in range(len(input_ids)):
            each_word_rel = [49] * len(cls2objindex[cls_])
            each_word_rel += [(0 if ii == word_i else
                get_position_emb_index(abs(ii - word_i), right=ii > word_i)
                ) for ii in range(len(input_ids))]
            rel_position_list.append(each_word_rel)
        rel_position_np = np.array(rel_position_list, dtype=np.int64)
        assert rel_position_np.shape[0] == rel_position_np.shape[1]
        rel_position.append(rel_position_np)
        sub_span = [self.img_index] * len(cls2objindex[cls_]) + input_ids
        encoder_input_ids += sub_span
        encoder_img_mask += [1] * len(cls2objindex[cls_]) + [0] * len(input_ids
            )
        encoder_cls += [cls_] * len(sub_span)
        img_order += [o[0] for o in cls2objindex[cls_]]
    encoder_input_ids.append(self.tokenizer.eos_token_id)
    encoder_img_mask.append(0)
    encoder_cls.append(0)
    dim_shape = sum([r.shape[0] for r in rel_position])
    encoder_rel_position_np = np.ones((dim_shape + 1, dim_shape + 1), dtype
        =np.int64) * 54
    if not self.config.use_orginal_enc_pos_embs:
        accumulate_dim = 0
        rel_start_position = []
        for r in rel_position:
            encoder_rel_position_np[accumulate_dim:accumulate_dim + r.shape
                [0], accumulate_dim:accumulate_dim + r.shape[0]] = r
            rel_start_position.append(accumulate_dim)
            accumulate_dim += r.shape[0]
        encoder_rel_position_np[-1, -1] = 0
        for i, ri in enumerate(rel_position):
            for j, rj in enumerate(rel_position):
                if i == j:
                    continue
                i_vis_end = len(cls2objindex[key_order[i]])
                j_vis_end = len(cls2objindex[key_order[j]])
                for i_index in range(ri.shape[0]):
                    for j_index in range(rj.shape[0]):
                        if i_index < i_vis_end and j_index < j_vis_end:
                            encoder_rel_position_np[rel_start_position[i] +
                                i_index, rel_start_position[j] + j_index] = 50
                        elif i_index < i_vis_end and j_index >= j_vis_end:
                            encoder_rel_position_np[rel_start_position[i] +
                                i_index, rel_start_position[j] + j_index] = 51
                        elif i_index >= i_vis_end and j_index < j_vis_end:
                            encoder_rel_position_np[rel_start_position[i] +
                                i_index, rel_start_position[j] + j_index] = 52
                        elif i_index >= i_vis_end and j_index >= j_vis_end:
                            encoder_rel_position_np[rel_start_position[i] +
                                i_index, rel_start_position[j] + j_index] = 53
    obj_feature_np = np.zeros((len(encoder_input_ids), obj_features.shape[-
        1]), dtype=np.float32)
    obj_box_np = np.zeros((len(encoder_input_ids), obj_boxes.shape[-1]),
        dtype=np.float32)
    obj_index = 0
    for i, m in enumerate(encoder_img_mask):
        if m == 1:
            if img_order[obj_index] < obj_size:
                cur_index = img_order[obj_index]
                obj_feature_np[i] = obj_features[cur_index]
                obj_box_np[i] = obj_boxes[cur_index]
            else:
                cur_index = img_order[obj_index] - obj_size
                obj_feature_np[i] = copy_obj_features[cur_index]
                obj_box_np[i] = copy_obj_boxes[cur_index]
            obj_index += 1
    self.obj_cache[img_id] = {'encoder_rel_position':
        encoder_rel_position_np, 'encoder_input_ids': np.array(
        encoder_input_ids, dtype=np.int64), 'encoder_cls': np.array(
        encoder_cls, dtype=np.int64), 'encoder_img_mask': np.array(
        encoder_img_mask, dtype=np.float32), 'obj_feature_np':
        obj_feature_np, 'obj_box_np': obj_box_np, 'image_id': img_id}
