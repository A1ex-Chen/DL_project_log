def _predict_kitti_to_file(net, example, result_save_path, class_names,
    center_limit_range=None, lidar_input=False):
    batch_image_shape = example['image_shape']
    batch_imgidx = example['image_idx']
    predictions_dicts = net(example)
    for i, preds_dict in enumerate(predictions_dicts):
        image_shape = batch_image_shape[i]
        img_idx = preds_dict['image_idx']
        if preds_dict['bbox'] is not None:
            box_2d_preds = preds_dict['bbox'].data.cpu().numpy()
            box_preds = preds_dict['box3d_camera'].data.cpu().numpy()
            scores = preds_dict['scores'].data.cpu().numpy()
            box_preds_lidar = preds_dict['box3d_lidar'].data.cpu().numpy()
            box_preds = box_preds[:, [0, 1, 2, 4, 5, 3, 6]]
            label_preds = preds_dict['label_preds'].data.cpu().numpy()
            result_lines = []
            for box, box_lidar, bbox, score, label in zip(box_preds,
                box_preds_lidar, box_2d_preds, scores, label_preds):
                if not lidar_input:
                    if bbox[0] > image_shape[1] or bbox[1] > image_shape[0]:
                        continue
                    if bbox[2] < 0 or bbox[3] < 0:
                        continue
                if center_limit_range is not None:
                    limit_range = np.array(center_limit_range)
                    if np.any(box_lidar[:3] < limit_range[:3]) or np.any(
                        box_lidar[:3] > limit_range[3:]):
                        continue
                bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                bbox[:2] = np.maximum(bbox[:2], [0, 0])
                result_dict = {'name': class_names[int(label)], 'alpha': -
                    np.arctan2(-box_lidar[1], box_lidar[0]) + box[6],
                    'bbox': bbox, 'location': box[:3], 'dimensions': box[3:
                    6], 'rotation_y': box[6], 'score': score}
                result_line = kitti.kitti_result_line(result_dict)
                result_lines.append(result_line)
        else:
            result_lines = []
        result_file = (
            f'{result_save_path}/{kitti.get_image_index_str(img_idx)}.txt')
        result_str = '\n'.join(result_lines)
        with open(result_file, 'w') as f:
            f.write(result_str)
