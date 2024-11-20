def validate(model, valid_loader, device, cameras, confusion_matrices,
    modality, loss_function_valid, logs, ckpt_dir, epoch,
    loss_function_valid_unweighted=None, add_log_key='', debug_mode=False):
    valid_split = valid_loader.dataset.split + add_log_key
    print(f'Validation on {valid_split}')
    validation_start_time = time.time()
    cm_time = 0
    forward_time = 0
    post_processing_time = 0
    copy_to_gpu_time = 0
    model.eval()
    miou = dict()
    ious = dict()
    pacc = dict()
    macc = dict()
    loss_function_valid.reset_loss()
    if loss_function_valid_unweighted is not None:
        loss_function_valid_unweighted.reset_loss()
    for camera in cameras:
        with valid_loader.dataset.filter_camera(camera):
            confusion_matrices[camera].reset_conf_matrix()
            print(f'{camera}: {len(valid_loader.dataset)} samples')
            for i, sample in enumerate(valid_loader):
                copy_to_gpu_time_start = time.time()
                if modality in ['rgbd', 'rgb']:
                    image = sample['image'].to(device)
                if modality in ['rgbd', 'depth']:
                    depth = sample['depth'].to(device)
                if not device.type == 'cpu':
                    torch.cuda.synchronize()
                copy_to_gpu_time += time.time() - copy_to_gpu_time_start
                with torch.no_grad():
                    forward_time_start = time.time()
                    if modality == 'rgbd':
                        prediction = model(image, depth)
                    elif modality == 'rgb':
                        prediction = model(image)
                    else:
                        prediction = model(depth)
                    if not device.type == 'cpu':
                        torch.cuda.synchronize()
                    forward_time += time.time() - forward_time_start
                    post_processing_time_start = time.time()
                    loss_function_valid.add_loss_of_batch(prediction,
                        sample['label'].to(device))
                    if loss_function_valid_unweighted is not None:
                        loss_function_valid_unweighted.add_loss_of_batch(
                            prediction, sample['label'].to(device))
                    label = sample['label_orig']
                    _, image_h, image_w = label.shape
                    prediction = F.interpolate(prediction, (image_h,
                        image_w), mode='bilinear', align_corners=False)
                    prediction = torch.argmax(prediction, dim=1)
                    mask = label > 0
                    label = torch.masked_select(label, mask)
                    prediction = torch.masked_select(prediction, mask.to(
                        device))
                    label -= 1
                    prediction = prediction.cpu()
                    label = label.numpy()
                    prediction = prediction.numpy()
                    post_processing_time += time.time(
                        ) - post_processing_time_start
                    cm_start_time = time.time()
                    confusion_matrices[camera].update_conf_matrix(label,
                        prediction)
                    cm_time += time.time() - cm_start_time
                    if debug_mode:
                        break
            cm_start_time = time.time()
            miou[camera], ious[camera], pacc[camera], macc[camera
                ] = confusion_matrices[camera].compute_miou()
            cm_time += time.time() - cm_start_time
            print(f'mIoU {valid_split} {camera}: {miou[camera]}')
            print(f'pacc {valid_split} {camera}: {pacc[camera]}')
            print(f'macc {valid_split} {camera}: {macc[camera]}')
    cm_start_time = time.time()
    confusion_matrices['all'].reset_conf_matrix()
    for camera in cameras:
        confusion_matrices['all'
            ].overall_confusion_matrix += confusion_matrices[camera
            ].overall_confusion_matrix
    miou['all'], ious['all'], pacc['all'], macc['all'] = confusion_matrices[
        'all'].compute_miou()
    cm_time += time.time() - cm_start_time
    print(f"mIoU {valid_split}: {miou['all']}")
    print(f"pacc {valid_split}: {pacc['all']}")
    print(f"macc {valid_split}: {macc['all']}")
    validation_time = time.time() - validation_start_time
    with open(os.path.join(ckpt_dir, 'confusion_matrices',
        f'cm_epoch_{epoch}.pickle'), 'wb') as f:
        pickle.dump({k: cm.overall_confusion_matrix for k, cm in
            confusion_matrices.items()}, f, protocol=pickle.HIGHEST_PROTOCOL)
    logs[f'loss_{valid_split}'] = loss_function_valid.compute_whole_loss()
    if loss_function_valid_unweighted is not None:
        logs[f'loss_{valid_split}_unweighted'
            ] = loss_function_valid_unweighted.compute_whole_loss()
    logs[f'mIoU_{valid_split}'] = miou['all']
    for camera in cameras:
        logs[f'mIoU_{valid_split}_{camera}'] = miou[camera]
    logs[f'pacc_{valid_split}'] = pacc['all']
    for camera in cameras:
        logs[f'pacc_{valid_split}_{camera}'] = pacc[camera]
    logs[f'macc_{valid_split}'] = macc['all']
    for camera in cameras:
        logs[f'macc_{valid_split}_{camera}'] = macc[camera]
    logs['time_validation'] = validation_time
    logs['time_confusion_matrix'] = cm_time
    logs['time_forward'] = forward_time
    logs['time_post_processing'] = post_processing_time
    logs['time_copy_to_gpu'] = copy_to_gpu_time
    for i, iou_value in enumerate(ious['all']):
        logs[f'IoU_{valid_split}_class_{i}'] = iou_value
    return miou, logs, pacc, macc
