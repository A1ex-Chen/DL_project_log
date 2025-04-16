def parse_predictions(predicted_boxes, sem_cls_probs, objectness_probs,
    point_cloud, config_dict):
    """Parse predictions to OBB parameters and suppress overlapping boxes

    Args:
        end_points: dict
            {point_clouds, center, heading_scores, heading_residuals,
            size_scores, size_residuals, sem_cls_scores}
        config_dict: dict
            {dataset_config, remove_empty_box, use_3d_nms, nms_iou,
            use_old_type_nms, conf_thresh, per_class_proposal}

    Returns:
        batch_pred_map_cls: a list of len == batch size (BS)
            [pred_list_i], i = 0, 1, ..., BS-1
            where pred_list_i = [(pred_sem_cls, box_params, box_score)_j]
            where j = 0, ..., num of valid detections - 1 from sample input i
    """
    sem_cls_probs = sem_cls_probs.detach().cpu().numpy()
    pred_sem_cls_prob = np.max(sem_cls_probs, -1)
    pred_sem_cls = np.argmax(sem_cls_probs, -1)
    obj_prob = objectness_probs.detach().cpu().numpy()
    pred_corners_3d_upright_camera = predicted_boxes.detach().cpu().numpy()
    K = pred_corners_3d_upright_camera.shape[1]
    bsize = pred_corners_3d_upright_camera.shape[0]
    nonempty_box_mask = np.ones((bsize, K))
    if config_dict['remove_empty_box']:
        batch_pc = point_cloud.cpu().numpy()[:, :, 0:3]
        for i in range(bsize):
            pc = batch_pc[i, :, :]
            for j in range(K):
                box3d = pred_corners_3d_upright_camera[i, j, :, :]
                box3d = flip_axis_to_depth(box3d)
                pc_in_box, inds = extract_pc_in_box3d(pc, box3d)
                if len(pc_in_box) < 5:
                    nonempty_box_mask[i, j] = 0
            if nonempty_box_mask[i].sum() == 0:
                nonempty_box_mask[i, obj_prob[i].argmax()] = 1
    if 'no_nms' in config_dict and config_dict['no_nms']:
        pred_mask = nonempty_box_mask
    elif not config_dict['use_3d_nms']:
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_2d_with_prob = np.zeros((K, 5))
            for j in range(K):
                boxes_2d_with_prob[j, 0] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_2d_with_prob[j, 2] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_2d_with_prob[j, 1] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_2d_with_prob[j, 3] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_2d_with_prob[j, 4] = obj_prob[i, j]
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            assert len(nonempty_box_inds) > 0
            pick = nms_2d_faster(boxes_2d_with_prob[nonempty_box_mask[i, :] ==
                1, :], config_dict['nms_iou'], config_dict['use_old_type_nms'])
            assert len(pick) > 0
            pred_mask[i, nonempty_box_inds[pick]] = 1
    elif config_dict['use_3d_nms'] and not config_dict['cls_nms']:
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K, 7))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 1] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 2] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 3] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 4] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 5] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 6] = obj_prob[i, j]
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            assert len(nonempty_box_inds) > 0
            pick = nms_3d_faster(boxes_3d_with_prob[nonempty_box_mask[i, :] ==
                1, :], config_dict['nms_iou'], config_dict['use_old_type_nms'])
            assert len(pick) > 0
            pred_mask[i, nonempty_box_inds[pick]] = 1
    elif config_dict['use_3d_nms'] and config_dict['cls_nms']:
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K, 8))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 1] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 2] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 3] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 4] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 5] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 6] = obj_prob[i, j]
                boxes_3d_with_prob[j, 7] = pred_sem_cls[i, j]
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            assert len(nonempty_box_inds) > 0
            pick = nms_3d_faster_samecls(boxes_3d_with_prob[
                nonempty_box_mask[i, :] == 1, :], config_dict['nms_iou'],
                config_dict['use_old_type_nms'])
            assert len(pick) > 0
            pred_mask[i, nonempty_box_inds[pick]] = 1
    batch_pred_map_cls = []
    for i in range(bsize):
        if config_dict['per_class_proposal']:
            assert config_dict['use_cls_confidence_only'] is False
            cur_list = []
            for ii in range(config_dict['dataset_config'].num_semcls):
                cur_list += [(ii, pred_corners_3d_upright_camera[i, j], 
                    sem_cls_probs[i, j, ii] * obj_prob[i, j]) for j in
                    range(pred_corners_3d_upright_camera.shape[1]) if 
                    pred_mask[i, j] == 1 and obj_prob[i, j] > config_dict[
                    'conf_thresh']]
            batch_pred_map_cls.append(cur_list)
        elif config_dict['use_cls_confidence_only']:
            batch_pred_map_cls.append([(pred_sem_cls[i, j].item(),
                pred_corners_3d_upright_camera[i, j], sem_cls_probs[i, j,
                pred_sem_cls[i, j].item()]) for j in range(
                pred_corners_3d_upright_camera.shape[1]) if pred_mask[i, j] ==
                1 and obj_prob[i, j] > config_dict['conf_thresh']])
        else:
            batch_pred_map_cls.append([(pred_sem_cls[i, j].item(),
                pred_corners_3d_upright_camera[i, j], obj_prob[i, j]) for j in
                range(pred_corners_3d_upright_camera.shape[1]) if pred_mask
                [i, j] == 1 and obj_prob[i, j] > config_dict['conf_thresh']])
    return batch_pred_map_cls
