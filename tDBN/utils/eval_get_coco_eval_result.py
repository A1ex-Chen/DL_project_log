def get_coco_eval_result(gt_annos, dt_annos, current_classes):
    class_to_name = {(0): 'Car', (1): 'Pedestrian', (2): 'Cyclist', (3):
        'Van', (4): 'Person_sitting', (5): 'car', (6): 'tractor', (7):
        'trailer'}
    class_to_range = {(0): [0.5, 1.0, 0.05], (1): [0.25, 0.75, 0.05], (2):
        [0.25, 0.75, 0.05], (3): [0.5, 1.0, 0.05], (4): [0.25, 0.75, 0.05],
        (5): [0.5, 1.0, 0.05], (6): [0.5, 1.0, 0.05], (7): [0.5, 1.0, 0.05]}
    class_to_range = {(0): [0.5, 0.95, 10], (1): [0.25, 0.7, 10], (2): [
        0.25, 0.7, 10], (3): [0.5, 0.95, 10], (4): [0.25, 0.7, 10], (5): [
        0.5, 0.95, 10], (6): [0.5, 0.95, 10], (7): [0.5, 0.95, 10]}
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    overlap_ranges = np.zeros([3, 3, len(current_classes)])
    for i, curcls in enumerate(current_classes):
        overlap_ranges[:, :, i] = np.array(class_to_range[curcls])[:, np.
            newaxis]
    result = ''
    compute_aos = False
    for anno in dt_annos:
        if anno['alpha'].shape[0] != 0:
            if anno['alpha'][0] != -10:
                compute_aos = True
            break
    mAPbbox, mAPbev, mAP3d, mAPaos = do_coco_style_eval(gt_annos, dt_annos,
        current_classes, overlap_ranges, compute_aos)
    for j, curcls in enumerate(current_classes):
        o_range = np.array(class_to_range[curcls])[[0, 2, 1]]
        o_range[1] = (o_range[2] - o_range[0]) / (o_range[1] - 1)
        result += print_str(
            f'{class_to_name[curcls]} coco AP@{{:.2f}}:{{:.2f}}:{{:.2f}}:'.
            format(*o_range))
        result += print_str(
            f'bbox AP:{mAPbbox[j, 0]:.2f}, {mAPbbox[j, 1]:.2f}, {mAPbbox[j, 2]:.2f}'
            )
        result += print_str(
            f'bev  AP:{mAPbev[j, 0]:.2f}, {mAPbev[j, 1]:.2f}, {mAPbev[j, 2]:.2f}'
            )
        result += print_str(
            f'3d   AP:{mAP3d[j, 0]:.2f}, {mAP3d[j, 1]:.2f}, {mAP3d[j, 2]:.2f}')
        if compute_aos:
            result += print_str(
                f'aos  AP:{mAPaos[j, 0]:.2f}, {mAPaos[j, 1]:.2f}, {mAPaos[j, 2]:.2f}'
                )
    return result
