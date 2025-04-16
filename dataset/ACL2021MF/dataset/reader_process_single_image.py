def process_single_image(self, image_id, i, boxes_h5):
    feature = boxes_h5['features'][i].reshape(-1, 2048)
    box_np = boxes_h5['boxes'][i].reshape(-1, 4)
    box_score = boxes_h5['scores'][i]
    class_list = (boxes_h5['classes'][i] + self.cls_start_index).tolist()
    new_box_np = np.zeros((box_np.shape[0], 8))
    if box_np.shape[0] > 0:
        box_np[:, 0] /= self._width[i]
        box_np[:, 2] /= self._width[i]
        box_np[:, 1] /= self._height[i]
        box_np[:, 3] /= self._height[i]
        new_box_np[:, :4] = box_np
        new_box_np[:, 4] = box_np[:, 2] - box_np[:, 0]
        new_box_np[:, 5] = box_np[:, 3] - box_np[:, 1]
        new_box_np[:, 6] = (box_np[:, 2] - box_np[:, 0]) * (box_np[:, 3] -
            box_np[:, 1])
        min_size = min(box_score.shape[0], box_np.shape[0])
        new_box_np[:min_size, 7] = box_score[:min_size]
    feature = feature[:box_np.shape[0]]
    new_box_np = new_box_np[:box_np.shape[0]]
    box_score = box_score[:box_np.shape[0]]
    class_list = class_list[:box_np.shape[0]]
    if not self.variant_copy_candidates:
        _class = []
        _box = []
        _feature = []
        min_size = min(feature.shape[0], new_box_np.shape[0])
        feature = feature[:min_size]
        new_box_np = new_box_np[:min_size]
        box_score = box_score[:min_size]
        class_list = class_list[:min_size]
        for idx, box_cls in enumerate(class_list):
            if box_cls not in self.detection_dict or box_score[idx
                ] < self.min_score:
                continue
            _box.append(new_box_np[idx])
            _class.append(box_cls)
            _feature.append(feature[idx])
        new_box_np = np.zeros((len(_box), 8))
        for i, bb in enumerate(_box):
            new_box_np[i] = bb
        feature_np = np.array(_feature)
        class_np = np.array(_class)
        obj_mask = np.ones((len(_class),), dtype=np.float32)
        for idx, box_cls in enumerate(class_np):
            text_class = self.detection_dict[box_cls]
            if text_class in self._blacklist_categories:
                obj_mask[idx] = 0.0
        if self.object_filtering:
            keep = nms(new_box_np, [self.detection_dict[box_cls] for
                box_cls in _class], self.hierarchy_finder.class_structure)
            for idx in range(len(_class)):
                if idx not in keep:
                    obj_mask[idx] = 0.0
        if new_box_np.shape[0] > 0:
            anns = []
            for idx, (box, cls_, mask) in enumerate(zip(new_box_np,
                class_np, obj_mask)):
                if mask == 1.0:
                    anns.append((box, cls_, idx))
            anns = sorted(anns, key=lambda x: x[0][7], reverse=True)
            if self.object_filtering:
                anns = anns[:self.top_k]
            seen_class = []
            for box, cls_, idx in anns:
                if cls_ not in seen_class:
                    seen_class.append(cls_)
                    obj_mask[idx] = 2.0
            obj_mask[obj_mask < 2.0] = 0.0
            obj_mask[obj_mask == 2.0] = 1.0
        class_list = class_np.tolist()
        text_class = [self.detection_dict[v] for v in class_list]
        if self.hierarchy_finder is not None:
            parent_class = [self.hierarchy_finder.find_parent(v) for v in
                text_class]
            parent_class_index = [self.hierarchy_finder.abstract_list[v] for
                v in parent_class]
        else:
            parent_class_index = [(0) for _ in range(len(text_class))]
        new_box_np = new_box_np.astype('float32')
        class_np = np.array(class_list).astype('int64')
        feature_np = feature_np.astype('float32')
        parent_class_np = np.array(parent_class_index).astype('int64')
        if not self.copy_candidate_clear_up:
            self.cache[image_id] = {'predicted_boxes': new_box_np,
                'predicted_classes': class_np, 'predicted_feature':
                feature_np, 'parent_classes': parent_class_np,
                'predicted_mask': obj_mask}
        else:
            self.cache[image_id] = {'predicted_boxes': new_box_np[obj_mask ==
                1], 'predicted_classes': class_np[obj_mask == 1],
                'predicted_feature': feature_np[obj_mask == 1],
                'parent_classes': parent_class_np[obj_mask == 1],
                'predicted_mask': obj_mask[obj_mask == 1]}
    else:
        text_class = [self.detection_dict[v] for v in class_list]
        if self.hierarchy_finder is not None:
            parent_class = [self.hierarchy_finder.find_parent(v) for v in
                text_class]
            parent_class_index = [self.hierarchy_finder.abstract_list[v] for
                v in parent_class]
        else:
            parent_class_index = [(0) for _ in range(len(text_class))]
        new_box_np = new_box_np.astype('float32')
        class_np = np.array(class_list).astype('int64')
        feature_np = feature.astype('float32')
        parent_class_np = np.array(parent_class_index).astype('int64')
        obj_mask = np.ones_like(class_np)
        if not self.copy_candidate_clear_up:
            self.cache[image_id] = {'predicted_boxes': new_box_np,
                'predicted_classes': class_np, 'predicted_feature':
                feature_np, 'parent_classes': parent_class_np,
                'predicted_mask': obj_mask}
        else:
            self.cache[image_id] = {'predicted_boxes': new_box_np[obj_mask ==
                1], 'predicted_classes': class_np[obj_mask == 1],
                'predicted_feature': feature_np[obj_mask == 1],
                'parent_classes': parent_class_np[obj_mask == 1],
                'predicted_mask': obj_mask[obj_mask == 1]}
