def clean_data(gt_anno, dt_anno, current_class, difficulty):
    CLASS_NAMES = ['car', 'pedestrian', 'cyclist', 'van', 'person_sitting',
        'car', 'tractor', 'trailer']
    MIN_HEIGHT = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.3, 0.5]
    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = CLASS_NAMES[current_class].lower()
    num_gt = len(gt_anno['name'])
    num_dt = len(dt_anno['name'])
    num_valid_gt = 0
    for i in range(num_gt):
        bbox = gt_anno['bbox'][i]
        gt_name = gt_anno['name'][i].lower()
        height = bbox[3] - bbox[1]
        valid_class = -1
        if gt_name == current_cls_name:
            valid_class = 1
        elif current_cls_name == 'Pedestrian'.lower(
            ) and 'Person_sitting'.lower() == gt_name:
            valid_class = 0
        elif current_cls_name == 'Car'.lower() and 'Van'.lower() == gt_name:
            valid_class = 0
        else:
            valid_class = -1
        ignore = False
        if gt_anno['occluded'][i] > MAX_OCCLUSION[difficulty] or gt_anno[
            'truncated'][i] > MAX_TRUNCATION[difficulty
            ] or height <= MIN_HEIGHT[difficulty]:
            ignore = True
        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif valid_class == 0 or ignore and valid_class == 1:
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)
        if gt_anno['name'][i] == 'DontCare':
            dc_bboxes.append(gt_anno['bbox'][i])
    for i in range(num_dt):
        if dt_anno['name'][i].lower() == current_cls_name:
            valid_class = 1
        else:
            valid_class = -1
        height = abs(dt_anno['bbox'][i, 3] - dt_anno['bbox'][i, 1])
        if height < MIN_HEIGHT[difficulty]:
            ignored_dt.append(1)
        elif valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)
    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes
