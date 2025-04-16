def predict(self, example, preds_dict):
    t = time.time()
    batch_size = example['anchors'].shape[0]
    batch_anchors = example['anchors'].view(batch_size, -1, 7)
    self._total_inference_count += batch_size
    batch_rect = example['rect']
    batch_Trv2c = example['Trv2c']
    batch_P2 = example['P2']
    if 'anchors_mask' not in example:
        batch_anchors_mask = [None] * batch_size
    else:
        batch_anchors_mask = example['anchors_mask'].view(batch_size, -1)
    batch_imgidx = example['image_idx']
    self._total_forward_time += time.time() - t
    t = time.time()
    batch_box_preds = preds_dict['box_preds']
    batch_cls_preds = preds_dict['cls_preds']
    batch_box_preds = batch_box_preds.view(batch_size, -1, self._box_coder.
        code_size)
    num_class_with_bg = self._num_class
    if not self._encode_background_as_zeros:
        num_class_with_bg = self._num_class + 1
    batch_cls_preds = batch_cls_preds.view(batch_size, -1, num_class_with_bg)
    batch_box_preds = self._box_coder.decode_torch(batch_box_preds,
        batch_anchors)
    if self._use_direction_classifier:
        batch_dir_preds = preds_dict['dir_cls_preds']
        batch_dir_preds = batch_dir_preds.view(batch_size, -1, 2)
    else:
        batch_dir_preds = [None] * batch_size
    predictions_dicts = []
    for box_preds, cls_preds, dir_preds, rect, Trv2c, P2, img_idx, a_mask in zip(
        batch_box_preds, batch_cls_preds, batch_dir_preds, batch_rect,
        batch_Trv2c, batch_P2, batch_imgidx, batch_anchors_mask):
        if a_mask is not None:
            box_preds = box_preds[a_mask]
            cls_preds = cls_preds[a_mask]
        if self._use_direction_classifier:
            if a_mask is not None:
                dir_preds = dir_preds[a_mask]
            dir_labels = torch.max(dir_preds, dim=-1)[1]
        if self._encode_background_as_zeros:
            assert self._use_sigmoid_score is True
            total_scores = torch.sigmoid(cls_preds)
        elif self._use_sigmoid_score:
            total_scores = torch.sigmoid(cls_preds)[..., 1:]
        else:
            total_scores = F.softmax(cls_preds, dim=-1)[..., 1:]
        if self._use_rotate_nms:
            nms_func = box_torch_ops.rotate_nms
        else:
            nms_func = box_torch_ops.nms
        selected_boxes = None
        selected_labels = None
        selected_scores = None
        selected_dir_labels = None
        if self._multiclass_nms:
            boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
            if not self._use_rotate_nms:
                box_preds_corners = box_torch_ops.center_to_corner_box2d(
                    boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                    boxes_for_nms[:, 4])
                boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                    box_preds_corners)
            boxes_for_mcnms = boxes_for_nms.unsqueeze(1)
            selected_per_class = box_torch_ops.multiclass_nms(nms_func=
                nms_func, boxes=boxes_for_mcnms, scores=total_scores,
                num_class=self._num_class, pre_max_size=self.
                _nms_pre_max_size, post_max_size=self._nms_post_max_size,
                iou_threshold=self._nms_iou_threshold, score_thresh=self.
                _nms_score_threshold)
            selected_boxes, selected_labels, selected_scores = [], [], []
            selected_dir_labels = []
            for i, selected in enumerate(selected_per_class):
                if selected is not None:
                    num_dets = selected.shape[0]
                    selected_boxes.append(box_preds[selected])
                    selected_labels.append(torch.full([num_dets], i, dtype=
                        torch.int64))
                    if self._use_direction_classifier:
                        selected_dir_labels.append(dir_labels[selected])
                    selected_scores.append(total_scores[selected, i])
            if len(selected_boxes) > 0:
                selected_boxes = torch.cat(selected_boxes, dim=0)
                selected_labels = torch.cat(selected_labels, dim=0)
                selected_scores = torch.cat(selected_scores, dim=0)
                if self._use_direction_classifier:
                    selected_dir_labels = torch.cat(selected_dir_labels, dim=0)
            else:
                selected_boxes = None
                selected_labels = None
                selected_scores = None
                selected_dir_labels = None
        else:
            if num_class_with_bg == 1:
                top_scores = total_scores.squeeze(-1)
                top_labels = torch.zeros(total_scores.shape[0], device=
                    total_scores.device, dtype=torch.long)
            else:
                top_scores, top_labels = torch.max(total_scores, dim=-1)
            if self._nms_score_threshold > 0.0:
                thresh = torch.tensor([self._nms_score_threshold], device=
                    total_scores.device).type_as(total_scores)
                top_scores_keep = top_scores >= thresh
                top_scores = top_scores.masked_select(top_scores_keep)
            if top_scores.shape[0] != 0:
                if self._nms_score_threshold > 0.0:
                    box_preds = box_preds[top_scores_keep]
                    if self._use_direction_classifier:
                        dir_labels = dir_labels[top_scores_keep]
                    top_labels = top_labels[top_scores_keep]
                boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                if not self._use_rotate_nms:
                    box_preds_corners = box_torch_ops.center_to_corner_box2d(
                        boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                        boxes_for_nms[:, 4])
                    boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                        box_preds_corners)
                selected = nms_func(boxes_for_nms, top_scores, pre_max_size
                    =self._nms_pre_max_size, post_max_size=self.
                    _nms_post_max_size, iou_threshold=self._nms_iou_threshold)
            else:
                selected = None
            if selected is not None:
                selected_boxes = box_preds[selected]
                if self._use_direction_classifier:
                    selected_dir_labels = dir_labels[selected]
                selected_labels = top_labels[selected]
                selected_scores = top_scores[selected]
        if selected_boxes is not None:
            box_preds = selected_boxes
            scores = selected_scores
            label_preds = selected_labels
            if self._use_direction_classifier:
                if self._encode_rad_error_by_sin:
                    dir_labels = selected_dir_labels
                    opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.byte()
                    box_preds[..., -1] += torch.where(opp_labels, torch.
                        tensor(np.pi).type_as(box_preds), torch.tensor(0.0)
                        .type_as(box_preds))
                else:
                    dir_labels = selected_dir_labels
                    inds1 = (box_preds[..., -1] > 0) & (dir_labels.byte() < 0.5
                        )
                    box_preds[inds1, -1] -= torch.tensor(np.pi).type_as(
                        box_preds)
                    inds2 = (box_preds[..., -1] < 0) & (dir_labels.byte() > 0.5
                        )
                    box_preds[inds2, -1] += torch.tensor(np.pi).type_as(
                        box_preds)
            final_box_preds = box_preds
            final_scores = scores
            final_labels = label_preds
            final_box_preds_camera = box_torch_ops.box_lidar_to_camera(
                final_box_preds, rect, Trv2c)
            locs = final_box_preds_camera[:, :3]
            dims = final_box_preds_camera[:, 3:6]
            angles = final_box_preds_camera[:, 6]
            camera_box_origin = [0.5, 1.0, 0.5]
            box_corners = box_torch_ops.center_to_corner_box3d(locs, dims,
                angles, camera_box_origin, axis=1)
            box_corners_in_image = box_torch_ops.project_to_image(box_corners,
                P2)
            minxy = torch.min(box_corners_in_image, dim=1)[0]
            maxxy = torch.max(box_corners_in_image, dim=1)[0]
            box_2d_preds = torch.cat([minxy, maxxy], dim=1)
            predictions_dict = {'bbox': box_2d_preds, 'box3d_camera':
                final_box_preds_camera, 'box3d_lidar': final_box_preds,
                'scores': final_scores, 'label_preds': label_preds,
                'image_idx': img_idx}
        else:
            predictions_dict = {'bbox': None, 'box3d_camera': None,
                'box3d_lidar': None, 'scores': None, 'label_preds': None,
                'image_idx': img_idx}
        predictions_dicts.append(predictions_dict)
    self._total_postprocess_time += time.time() - t
    return predictions_dicts
