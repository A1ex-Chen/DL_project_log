def train_main():
    args = parse_args()
    training_starttime = datetime.now().strftime('%d_%m_%Y-%H_%M_%S-%f')
    ckpt_dir = os.path.join(args.results_dir, args.dataset,
        f'checkpoints_{training_starttime}')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(ckpt_dir, 'confusion_matrices'), exist_ok=True)
    with open(os.path.join(ckpt_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)
    with open(os.path.join(ckpt_dir, 'argsv.txt'), 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')
    label_downsampling_rates = [8, 16, 32]
    data_loaders = prepare_data(args, ckpt_dir)
    if args.valid_full_res:
        train_loader, valid_loader, valid_loader_full_res = data_loaders
    else:
        train_loader, valid_loader = data_loaders
        valid_loader_full_res = None
    cameras = train_loader.dataset.cameras
    n_classes_without_void = train_loader.dataset.n_classes_without_void
    if args.class_weighting != 'None':
        class_weighting = train_loader.dataset.compute_class_weights(
            weight_mode=args.class_weighting, c=args.
            c_for_logarithmic_weighting)
    else:
        class_weighting = np.ones(n_classes_without_void)
    model, device = build_model(args, n_classes=n_classes_without_void)
    if args.freeze > 0:
        print('Freeze everything but the output layer(s).')
        for name, param in model.named_parameters():
            if 'out' not in name:
                param.requires_grad = False
    loss_function_train = utils.CrossEntropyLoss2d(weight=class_weighting,
        device=device)
    pixel_sum_valid_data = valid_loader.dataset.compute_class_weights(
        weight_mode='linear')
    pixel_sum_valid_data_weighted = np.sum(pixel_sum_valid_data *
        class_weighting)
    loss_function_valid = utils.CrossEntropyLoss2dForValidData(weight=
        class_weighting, weighted_pixel_sum=pixel_sum_valid_data_weighted,
        device=device)
    loss_function_valid_unweighted = (utils.
        CrossEntropyLoss2dForValidDataUnweighted(device=device))
    optimizer = get_optimizer(args, model)
    lr_scheduler = OneCycleLR(optimizer, max_lr=[i['lr'] for i in optimizer
        .param_groups], total_steps=args.epochs, div_factor=25, pct_start=
        0.1, anneal_strategy='cos', final_div_factor=10000.0)
    if args.last_ckpt:
        ckpt_path = os.path.join(ckpt_dir, args.last_ckpt)
        (epoch_last_ckpt, best_miou, best_miou_epoch, best_pacc,
            best_pacc_epoch, best_macc, best_macc_epoch) = load_ckpt(model,
            optimizer, ckpt_path, device)
        start_epoch = epoch_last_ckpt + 1
    else:
        start_epoch = 0
        best_miou = 0
        best_miou_epoch = 0
        best_pacc = 0
        best_pacc_epoch = 0
        best_macc = 0
        best_macc_epoch = 0
    valid_split = valid_loader.dataset.split
    log_keys = [f'mIoU_{valid_split}']
    if args.valid_full_res:
        log_keys.append(f'mIoU_{valid_split}_full-res')
        best_miou_full_res = 0
        best_pacc_full_res = 0
        best_macc_full_res = 0
    log_keys_for_csv = log_keys.copy()
    for camera in cameras:
        log_keys_for_csv.append(f'mIoU_{valid_split}_{camera}')
        log_keys_for_csv.append(f'pacc_{valid_split}_{camera}')
        log_keys_for_csv.append(f'macc_{valid_split}_{camera}')
        if args.valid_full_res:
            log_keys_for_csv.append(f'mIoU_{valid_split}_full-res_{camera}')
            log_keys_for_csv.append(f'pacc_{valid_split}_full-res_{camera}')
            log_keys_for_csv.append(f'macc_{valid_split}_full-res_{camera}')
    log_keys_for_csv.append(f'pacc_{valid_split}')
    log_keys_for_csv.append(f'macc_{valid_split}')
    log_keys_for_csv.append('epoch')
    for i in range(len(lr_scheduler.get_lr())):
        log_keys_for_csv.append('lr_{}'.format(i))
    log_keys_for_csv.extend(['loss_train_total', 'loss_train_full_size'])
    for rate in label_downsampling_rates:
        log_keys_for_csv.append('loss_train_down_{}'.format(rate))
    log_keys_for_csv.extend(['time_training', 'time_validation',
        'time_confusion_matrix', 'time_forward', 'time_post_processing',
        'time_copy_to_gpu'])
    valid_names = [valid_split]
    if args.valid_full_res:
        valid_names.append(valid_split + '_full-res')
    for valid_name in valid_names:
        for i in range(n_classes_without_void):
            log_keys_for_csv.append(f'IoU_{valid_name}_class_{i}')
        log_keys_for_csv.append(f'loss_{valid_name}')
        if loss_function_valid_unweighted is not None:
            log_keys_for_csv.append(f'loss_{valid_name}_unweighted')
    csvlogger = CSVLogger(log_keys_for_csv, os.path.join(ckpt_dir,
        'logs.csv'), append=True)
    confusion_matrices = dict()
    for camera in cameras:
        confusion_matrices[camera] = ConfusionMatrixTensorflow(
            n_classes_without_void)
        confusion_matrices['all'] = ConfusionMatrixTensorflow(
            n_classes_without_void)
    print('=====> computing network parameters and FLOPs')
    total_paramters = netParams(model)
    print('the number of parameters: %d ==> %.2f M' % (total_paramters, 
        total_paramters / 1000000.0))
    for epoch in range(int(start_epoch), args.epochs):
        if args.freeze == epoch and args.finetune is None:
            print('Unfreezing')
            for param in model.parameters():
                param.requires_grad = True
        logs = train_one_epoch(model, train_loader, device, optimizer,
            loss_function_train, epoch, lr_scheduler, args.modality,
            label_downsampling_rates, debug_mode=args.debug)
        miou, logs, macc, pacc = validate(model, valid_loader, device,
            cameras, confusion_matrices, args.modality, loss_function_valid,
            logs, ckpt_dir, epoch, loss_function_valid_unweighted,
            debug_mode=args.debug)
        if args.valid_full_res:
            miou_full_res, logs, macc_full_res, pacc_full_res = validate(model,
                valid_loader_full_res, device, cameras, confusion_matrices,
                args.modality, loss_function_valid, logs, ckpt_dir, epoch,
                loss_function_valid_unweighted, add_log_key='_full-res',
                debug_mode=args.debug)
        logs.pop('time', None)
        csvlogger.write_logs(logs)
        print(miou['all'])
        print(pacc['all'])
        print(macc['all'])
        save_current_checkpoint = False
        if miou['all'] > best_miou:
            best_miou = miou['all']
            best_miou_epoch = epoch
            save_current_checkpoint = True
        if pacc['all'] > best_pacc:
            best_pacc = pacc['all']
            best_pacc_epoch = epoch
            save_current_checkpoint = True
        if pacc['all'] > best_macc:
            best_macc = pacc['all']
            best_macc_epoch = epoch
            save_current_checkpoint = True
        if args.valid_full_res and miou_full_res['all'] > best_miou_full_res:
            best_miou_full_res = miou_full_res['all']
            best_miou_full_res_epoch = epoch
            save_current_checkpoint = True
        if args.valid_full_res and pacc_full_res['all'] > best_pacc_full_res:
            best_pacc_full_res = pacc_full_res['all']
            best_pacc_full_res_epoch = epoch
            save_current_checkpoint = True
        if args.valid_full_res and macc_full_res['all'] > best_macc_full_res:
            best_macc_full_res = macc_full_res['all']
            best_macc_full_res_epoch = epoch
            save_current_checkpoint = True
        if epoch >= 10 and save_current_checkpoint is True:
            save_ckpt(ckpt_dir, model, optimizer, epoch)
        save_ckpt_every_epoch(ckpt_dir, model, optimizer, epoch, best_miou,
            best_miou_epoch, best_pacc, best_pacc_epoch, best_macc,
            best_macc_epoch)
    with open(os.path.join(ckpt_dir, 'finished.txt'), 'w') as f:
        f.write('best miou: {}\n'.format(best_miou))
        f.write('best miou epoch: {}\n'.format(best_miou_epoch))
        f.write('best pacc: {}\n'.format(best_pacc))
        f.write('best pacc epoch: {}\n'.format(best_pacc_epoch))
        f.write('best macc: {}\n'.format(best_macc))
        f.write('best macc epoch: {}\n'.format(best_macc_epoch))
        if args.valid_full_res:
            f.write(f'best miou full res: {best_miou_full_res}\n')
            f.write(f'best miou full res epoch: {best_miou_full_res_epoch}\n')
            f.write(f'best pacc full res: {best_pacc_full_res}\n')
            f.write(f'best pacc full res epoch: {best_pacc_full_res_epoch}\n')
            f.write(f'best macc full res: {best_macc_full_res}\n')
            f.write(f'best macc full res epoch: {best_macc_full_res_epoch}\n')
    print('Training completed ')
