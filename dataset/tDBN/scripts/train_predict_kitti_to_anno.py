def predict_kitti_to_anno(net, example, class_names, center_limit_range=
    None, lidar_input=False, global_set=None):
    batch_image_shape = example['image_shape']
    batch_imgidx = example['image_idx']
    predictions_dicts = net(example)
    annos = []
    for i, preds_dict in enumerate(predictions_dicts):
        image_shape = batch_image_shape[i]
        img_idx = preds_dict['image_idx']
        if preds_dict['bbox'] is not None:
            box_2d_preds = preds_dict['bbox'].detach().cpu().numpy()
            box_preds = preds_dict['box3d_camera'].detach().cpu().numpy()
            scores = preds_dict['scores'].detach().cpu().numpy()
            box_preds_lidar = preds_dict['box3d_lidar'].detach().cpu().numpy()
            label_preds = preds_dict['label_preds'].detach().cpu().numpy()
            anno = kitti.get_start_result_anno()
            num_example = 0
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
                anno['name'].append(class_names[int(label)])
                anno['truncated'].append(0.0)
                anno['occluded'].append(0)
                anno['alpha'].append(-np.arctan2(-box_lidar[1], box_lidar[0
                    ]) + box[6])
                anno['bbox'].append(bbox)
                anno['dimensions'].append(box[3:6])
                anno['location'].append(box[:3])
                anno['rotation_y'].append(box[6])
                if global_set is not None:
                    for i in range(100000):
                        if score in global_set:
                            score -= 1 / 100000
                        else:
                            global_set.add(score)
                            break
                anno['score'].append(score)
                num_example += 1
            if num_example != 0:
                anno = {n: np.stack(v) for n, v in anno.items()}
                annos.append(anno)
            else:
                annos.append(kitti.empty_result_anno())
        else:
            annos.append(kitti.empty_result_anno())
        num_example = annos[-1]['name'].shape[0]
        annos[-1]['image_idx'] = np.array([img_idx] * num_example, dtype=np
            .int64)
    return annos
