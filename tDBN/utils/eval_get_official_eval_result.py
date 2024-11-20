def get_official_eval_result(gt_annos, dt_annos, current_classes,
    difficultys=[0, 1, 2]):
    overlap_0_7 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.7, 0.7, 0.7], [0.7,
        0.5, 0.5, 0.7, 0.5, 0.7, 0.7, 0.7], [0.7, 0.5, 0.5, 0.7, 0.5, 0.7, 
        0.7, 0.7]])
    overlap_0_5 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.5, 0.5, 0.5], [0.5,
        0.25, 0.25, 0.5, 0.25, 0.5, 0.5, 0.5], [0.5, 0.25, 0.25, 0.5, 0.25,
        0.5, 0.5, 0.5]])
    min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)
    class_to_name = {(0): 'Car', (1): 'Pedestrian', (2): 'Cyclist', (3):
        'Van', (4): 'Person_sitting', (5): 'car', (6): 'tractor', (7):
        'trailer'}
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
    min_overlaps = min_overlaps[:, :, current_classes]
    result = ''
    compute_aos = False
    for anno in dt_annos:
        if anno['alpha'].shape[0] != 0:
            if anno['alpha'][0] != -10:
                compute_aos = True
            break
    mAPbbox, mAPbev, mAP3d, mAPaos = do_eval_v2(gt_annos, dt_annos,
        current_classes, min_overlaps, compute_aos, difficultys)
    for j, curcls in enumerate(current_classes):
        for i in range(min_overlaps.shape[0]):
            result += print_str(
                f'{class_to_name[curcls]} AP@{{:.2f}}, {{:.2f}}, {{:.2f}}:'
                .format(*min_overlaps[i, :, j]))
            result += print_str(
                f'bbox AP:{mAPbbox[j, 0, i]:.2f}, {mAPbbox[j, 1, i]:.2f}, {mAPbbox[j, 2, i]:.2f}'
                )
            result += print_str(
                f'bev  AP:{mAPbev[j, 0, i]:.2f}, {mAPbev[j, 1, i]:.2f}, {mAPbev[j, 2, i]:.2f}'
                )
            result += print_str(
                f'3d   AP:{mAP3d[j, 0, i]:.2f}, {mAP3d[j, 1, i]:.2f}, {mAP3d[j, 2, i]:.2f}'
                )
            if compute_aos:
                result += print_str(
                    f'aos  AP:{mAPaos[j, 0, i]:.2f}, {mAPaos[j, 1, i]:.2f}, {mAPaos[j, 2, i]:.2f}'
                    )
    return result
