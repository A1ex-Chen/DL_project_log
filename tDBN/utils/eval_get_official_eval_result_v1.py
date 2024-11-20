def get_official_eval_result_v1(gt_annos, dt_annos, current_class):
    mAP_0_7 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5], [0.7, 0.5, 0.5, 0.7, 0.5
        ], [0.7, 0.5, 0.5, 0.7, 0.5]])
    mAP_0_5 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5], [0.5, 0.25, 0.25, 0.5, 
        0.25], [0.5, 0.25, 0.25, 0.5, 0.25]])
    mAP_list = [mAP_0_7, mAP_0_5]
    class_to_name = {(0): 'Car', (1): 'Pedestrian', (2): 'Cyclist', (3):
        'Van', (4): 'Person_sitting'}
    name_to_class = {v: n for n, v in class_to_name.items()}
    if isinstance(current_class, str):
        current_class = name_to_class[current_class]
    result = ''
    compute_aos = False
    for anno in dt_annos:
        if anno['alpha'].shape[0] != 0:
            if anno['alpha'][0] != -10:
                compute_aos = True
            break
    for mAP in mAP_list:
        mAPbbox, mAPbev, mAP3d, mAPaos = do_eval(gt_annos, dt_annos,
            current_class, mAP[:, current_class], compute_aos)
        result += print_str(
            f'{class_to_name[current_class]} AP@{{:.2f}}, {{:.2f}}, {{:.2f}}:'
            .format(*mAP[:, current_class]))
        result += print_str(
            f'bbox AP:{mAPbbox[0]:.2f}, {mAPbbox[1]:.2f}, {mAPbbox[2]:.2f}')
        result += print_str(
            f'bev  AP:{mAPbev[0]:.2f}, {mAPbev[1]:.2f}, {mAPbev[2]:.2f}')
        result += print_str(
            f'3d   AP:{mAP3d[0]:.2f}, {mAP3d[1]:.2f}, {mAP3d[2]:.2f}')
        if compute_aos:
            result += print_str(
                f'aos  AP:{mAPaos[0]:.2f}, {mAPaos[1]:.2f}, {mAPaos[2]:.2f}')
    return result
