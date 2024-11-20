def train(cfg, args, output_dir='', run_name=''):
    logger = logging.getLogger('xmuda.vfm.train')
    set_random_seed(cfg.RNG_SEED)
    cfg_mixed_data_xm_lambda = 0.05
    """
    Build Model A - Train with VFM-PL, SAM-Mix
    """
    if _enable_model_A_:
        model_type = 'A'
        print('Init Model ' + model_type)
        model_A_2d, train_metric_model_A_2d = build_model_2d(cfg, '-' +
            model_type)
        logger.info('Build 2D model:\n{}'.format(str(model_A_2d)))
        num_params = sum(param.numel() for param in model_A_2d.parameters())
        print('#Parameters: {:.2e}'.format(num_params))
        model_A_3d, train_metric_model_A_3d = build_model_3d(cfg, '-' +
            model_type)
        logger.info('Build 3D model:\n{}'.format(str(model_A_3d)))
        num_params = sum(param.numel() for param in model_A_3d.parameters())
        print('#Parameters: {:.2e}'.format(num_params))
        model_A_2d = model_A_2d.cuda()
        model_A_3d = model_A_3d.cuda()
        optimizer_model_A_2d = build_optimizer(cfg, model_A_2d)
        optimizer_model_A_3d = build_optimizer(cfg, model_A_3d)
        scheduler_model_A_2d = build_scheduler(cfg, optimizer_model_A_2d)
        scheduler_model_A_3d = build_scheduler(cfg, optimizer_model_A_3d)
        checkpointer_model_A_2d = CheckpointerV2(model_A_2d, optimizer=
            optimizer_model_A_2d, scheduler=scheduler_model_A_2d, save_dir=
            output_dir, logger=logger, postfix='_2d_A', max_to_keep=cfg.
            TRAIN.MAX_TO_KEEP)
        checkpoint_data_model_A_2d = checkpointer_model_A_2d.load(cfg.
            RESUME_PATH, resume=cfg.AUTO_RESUME, resume_states=cfg.
            RESUME_STATES)
        checkpointer_model_A_3d = CheckpointerV2(model_A_3d, optimizer=
            optimizer_model_A_3d, scheduler=scheduler_model_A_3d, save_dir=
            output_dir, logger=logger, postfix='_3d_A', max_to_keep=cfg.
            TRAIN.MAX_TO_KEEP)
        checkpoint_data_model_A_3d = checkpointer_model_A_3d.load(cfg.
            RESUME_PATH, resume=cfg.AUTO_RESUME, resume_states=cfg.
            RESUME_STATES)
        ckpt_period = cfg.TRAIN.CHECKPOINT_PERIOD
    """
    Build Model B - Train with VFM-PL, Cut-Mix
    """
    if _enable_model_B_:
        model_type = 'B'
        print('Init Model ' + model_type)
        model_B_2d, train_metric_model_B_2d = build_model_2d(cfg, '-' +
            model_type)
        logger.info('Build 2D model:\n{}'.format(str(model_B_2d)))
        num_params = sum(param.numel() for param in model_B_2d.parameters())
        print('#Parameters: {:.2e}'.format(num_params))
        model_B_3d, train_metric_model_B_3d = build_model_3d(cfg, '-' +
            model_type)
        logger.info('Build 3D model:\n{}'.format(str(model_B_3d)))
        num_params = sum(param.numel() for param in model_B_3d.parameters())
        print('#Parameters: {:.2e}'.format(num_params))
        model_B_2d = model_B_2d.cuda()
        model_B_3d = model_B_3d.cuda()
        optimizer_model_B_2d = build_optimizer(cfg, model_B_2d)
        optimizer_model_B_3d = build_optimizer(cfg, model_B_3d)
        scheduler_model_B_2d = build_scheduler(cfg, optimizer_model_B_2d)
        scheduler_model_B_3d = build_scheduler(cfg, optimizer_model_B_3d)
        checkpointer_model_B_2d = CheckpointerV2(model_B_2d, optimizer=
            optimizer_model_B_2d, scheduler=scheduler_model_B_2d, save_dir=
            output_dir, logger=logger, postfix='_2d_B', max_to_keep=cfg.
            TRAIN.MAX_TO_KEEP)
        checkpoint_data_model_B_2d = checkpointer_model_B_2d.load(cfg.
            RESUME_PATH, resume=cfg.AUTO_RESUME, resume_states=cfg.
            RESUME_STATES)
        checkpointer_model_B_3d = CheckpointerV2(model_B_3d, optimizer=
            optimizer_model_B_3d, scheduler=scheduler_model_B_3d, save_dir=
            output_dir, logger=logger, postfix='_3d_B', max_to_keep=cfg.
            TRAIN.MAX_TO_KEEP)
        checkpoint_data_model_B_3d = checkpointer_model_B_3d.load(cfg.
            RESUME_PATH, resume=cfg.AUTO_RESUME, resume_states=cfg.
            RESUME_STATES)
    max_iteration = cfg.SCHEDULER.MAX_ITERATION
    set_random_seed(cfg.RNG_SEED)
    train_dataloader_src = build_dataloader(cfg, args, mode='train', domain
        ='source', start_iteration=start_iteration)
    train_dataloader_trg = build_dataloader(cfg, args, mode='train', domain
        ='target', start_iteration=start_iteration)
    best_metric_name = 'best_{}'.format(cfg.VAL.METRIC)
    if _enable_model_A_:
        start_iteration = checkpoint_data_model_A_2d.get('iteration', 0)
        best_metric = {'2d': checkpoint_data_model_A_2d.get(
            best_metric_name, None), '3d': checkpoint_data_model_A_3d.get(
            best_metric_name, None)}
    elif _enable_model_B_:
        start_iteration = checkpoint_data_model_B_2d.get('iteration', 0)
        best_metric = {'2d': checkpoint_data_model_B_2d.get(
            best_metric_name, None), '3d': checkpoint_data_model_B_3d.get(
            best_metric_name, None)}
    else:
        return
    best_metric_iter = {'2d': -1, '3d': -1, 'ensemble': -1}
    logger.info('Start training from iteration {}'.format(start_iteration))

    def setup_train(model):
        if 'A' == model:
            model_A_2d.train()
            model_A_3d.train()
            train_metric_logger_A.reset()
        elif 'B' == model:
            model_B_2d.train()
            model_B_3d.train()
        else:
            raise ValueError('Unsupported type of Model Experiment Setting')
    if cfg.TRAIN.CLASS_WEIGHTS:
        class_weights = torch.tensor(cfg.TRAIN.CLASS_WEIGHTS).cuda()
    else:
        class_weights = None
    if cfg.TRAIN.CLASS_WEIGHTS_PL:
        class_weights_pl = torch.tensor(cfg.TRAIN.CLASS_WEIGHTS_PL).cuda()
    else:
        class_weights_pl = None
    if _enable_model_A_:
        train_metric_logger_A = init_metric_logger([train_metric_model_A_2d,
            train_metric_model_A_3d])
        val_metric_logger_A = MetricLogger(delimiter='  ')
        setup_train('A')
    if _enable_model_B_:
        train_metric_logger_B = init_metric_logger([train_metric_model_B_2d,
            train_metric_model_B_3d])
        val_metric_logger_B = MetricLogger(delimiter='  ')
    setup_train('B')
    end = time.time()
    train_iter_src = enumerate(train_dataloader_src)
    train_iter_trg = enumerate(train_dataloader_trg)
    for iteration in range(start_iteration, max_iteration):
        _, data_batch_src = train_iter_src.__next__()
        _, data_batch_trg = train_iter_trg.__next__()
        if ('SCN' in cfg.DATASET_SOURCE.TYPE and 'SCN' in cfg.
            DATASET_TARGET.TYPE):
            masks_src_seem = data_batch_src['masks_seem'].cuda()
            masks_trg_seem = data_batch_trg['masks_seem'].cuda()
            for id in range(cfg.TRAIN.BATCH_SIZE):
                """SAM Mix"""
                data_batch_src['sam_mix_image'][id][data_batch_trg[
                    'sampled_sam_mask'][id]] = data_batch_trg['img'][id][
                    data_batch_trg['sampled_sam_mask'][id]]
                data_batch_trg['sam_mix_image'][id][data_batch_src[
                    'sampled_sam_mask'][id]] = data_batch_src['img'][id][
                    data_batch_src['sampled_sam_mask'][id]]
                del_slice_src = data_batch_trg['sampled_sam_mask'][id][
                    data_batch_src['img_indices'][id][:, 0], data_batch_src
                    ['img_indices'][id][:, 1]] == False
                data_batch_src['sam_mix_label_2d'][id] = torch.cat((
                    data_batch_src['sam_mix_label_2d'][id][del_slice_src],
                    data_batch_trg['sam_pseudo_label_2d'][id].clone()), 0)
                data_batch_src['sam_mix_label_3d'][id] = torch.cat((
                    data_batch_src['sam_mix_label_3d'][id][del_slice_src],
                    data_batch_trg['sam_pseudo_label_3d'][id].clone()), 0)
                del_slice_trg = data_batch_src['sampled_sam_mask'][id][
                    data_batch_trg['img_indices'][id][:, 0], data_batch_trg
                    ['img_indices'][id][:, 1]] == False
                data_batch_trg['sam_mix_pseudo_label_2d'][id] = torch.cat((
                    data_batch_trg['sam_mix_pseudo_label_2d'][id][
                    del_slice_trg], data_batch_src['sam_label'][id].clone()), 0
                    )
                data_batch_trg['sam_mix_pseudo_label_3d'][id] = torch.cat((
                    data_batch_trg['sam_mix_pseudo_label_3d'][id][
                    del_slice_trg], data_batch_src['sam_label'][id].clone()), 0
                    )
                data_batch_src['sam_mix_indices'][id] = np.concatenate((
                    data_batch_src['sam_mix_indices'][id][del_slice_src],
                    data_batch_trg['img_indices'][id][data_batch_trg[
                    'sampled_sam_sel_indices'][id]]), axis=0)
                data_batch_trg['sam_mix_indices'][id] = np.concatenate((
                    data_batch_trg['sam_mix_indices'][id][del_slice_trg],
                    data_batch_src['img_indices'][id][data_batch_src[
                    'sampled_sam_sel_indices'][id]]), axis=0)
                copy_data_batch_src_sam_mix_x_0 = data_batch_src['sam_mix_x'][0
                    ][id].clone()
                copy_data_batch_src_sam_mix_x_1 = data_batch_src['sam_mix_x'][1
                    ][id].clone()
                if data_batch_src['sam_mix_x'][0][id].shape[0
                    ] != data_batch_src['sam_mix_x'][1][id].shape[0]:
                    alignment_error_count += 1
                    skip_batch = True
                    logger.info('Skip mis-alignment, total {}'.format(
                        alignment_error_count))
                    continue
                data_batch_src['sam_mix_x'][0][id] = data_batch_src['sam_mix_x'
                    ][0][id][del_slice_src]
                data_batch_src['sam_mix_x'][1][id] = data_batch_src['sam_mix_x'
                    ][1][id][del_slice_src]
                data_batch_src['sam_mix_x'][0][id] = torch.cat((
                    data_batch_src['sam_mix_x'][0][id], data_batch_trg[
                    'sam_mix_x'][0][id].clone()[data_batch_trg[
                    'sampled_sam_sel_indices'][id]]), 0)
                data_batch_src['sam_mix_x'][1][id] = torch.cat((
                    data_batch_src['sam_mix_x'][1][id], data_batch_trg[
                    'sam_mix_x'][1][id].clone()[data_batch_trg[
                    'sampled_sam_sel_indices'][id]]), 0)
                data_batch_trg['sam_mix_x'][0][id] = data_batch_trg['sam_mix_x'
                    ][0][id][del_slice_trg]
                data_batch_trg['sam_mix_x'][1][id] = data_batch_trg['sam_mix_x'
                    ][1][id][del_slice_trg]
                data_batch_trg['sam_mix_x'][0][id] = torch.cat((
                    data_batch_trg['sam_mix_x'][0][id],
                    copy_data_batch_src_sam_mix_x_0[data_batch_src[
                    'sampled_sam_sel_indices'][id]]), 0)
                data_batch_trg['sam_mix_x'][1][id] = torch.cat((
                    data_batch_trg['sam_mix_x'][1][id],
                    copy_data_batch_src_sam_mix_x_1[data_batch_src[
                    'sampled_sam_sel_indices'][id]]), 0)
                """Cut Mix"""
                data_batch_src['cut_mix_image'][id][data_batch_trg[
                    'cut_mask'][id]] = data_batch_trg['img'][id][data_batch_trg
                    ['cut_mask'][id]]
                data_batch_trg['cut_mix_image'][id][data_batch_src[
                    'cut_mask'][id]] = data_batch_src['img'][id][data_batch_src
                    ['cut_mask'][id]]
                del_slice_src = data_batch_trg['cut_mask'][id][
                    data_batch_src['img_indices'][id][:, 0], data_batch_src
                    ['img_indices'][id][:, 1]] == False
                data_batch_src['cut_mix_label_2d'][id] = torch.cat((
                    data_batch_src['cut_mix_label_2d'][id][del_slice_src],
                    data_batch_trg['cut_pseudo_label_2d'][id].clone()), 0)
                data_batch_src['cut_mix_label_3d'][id] = torch.cat((
                    data_batch_src['cut_mix_label_3d'][id][del_slice_src],
                    data_batch_trg['cut_pseudo_label_3d'][id].clone()), 0)
                del_slice_trg = data_batch_src['cut_mask'][id][
                    data_batch_trg['img_indices'][id][:, 0], data_batch_trg
                    ['img_indices'][id][:, 1]] == False
                data_batch_trg['cut_mix_pseudo_label_2d'][id] = torch.cat((
                    data_batch_trg['cut_mix_pseudo_label_2d'][id][
                    del_slice_trg], data_batch_src['cut_label'][id].clone()), 0
                    )
                data_batch_trg['cut_mix_pseudo_label_3d'][id] = torch.cat((
                    data_batch_trg['cut_mix_pseudo_label_3d'][id][
                    del_slice_trg], data_batch_src['cut_label'][id].clone()), 0
                    )
                data_batch_src['cut_mix_indices'][id] = np.concatenate((
                    data_batch_src['cut_mix_indices'][id][del_slice_src],
                    data_batch_trg['img_indices'][id][data_batch_trg[
                    'cut_sel_indices'][id]]), axis=0)
                data_batch_trg['cut_mix_indices'][id] = np.concatenate((
                    data_batch_trg['cut_mix_indices'][id][del_slice_trg],
                    data_batch_src['img_indices'][id][data_batch_src[
                    'cut_sel_indices'][id]]), axis=0)
                copy_data_batch_src_cut_mix_x_0 = data_batch_src['cut_mix_x'][0
                    ][id].clone()
                copy_data_batch_src_cut_mix_x_1 = data_batch_src['cut_mix_x'][1
                    ][id].clone()
                data_batch_src['cut_mix_x'][0][id] = data_batch_src['cut_mix_x'
                    ][0][id][del_slice_src]
                data_batch_src['cut_mix_x'][1][id] = data_batch_src['cut_mix_x'
                    ][1][id][del_slice_src]
                data_batch_src['cut_mix_x'][0][id] = torch.cat((
                    data_batch_src['cut_mix_x'][0][id], data_batch_trg[
                    'cut_mix_x'][0][id].clone()[data_batch_trg[
                    'cut_sel_indices'][id]]), 0)
                data_batch_src['cut_mix_x'][1][id] = torch.cat((
                    data_batch_src['cut_mix_x'][1][id], data_batch_trg[
                    'cut_mix_x'][1][id].clone()[data_batch_trg[
                    'cut_sel_indices'][id]]), 0)
                data_batch_trg['cut_mix_x'][0][id] = data_batch_trg['cut_mix_x'
                    ][0][id][del_slice_trg]
                data_batch_trg['cut_mix_x'][1][id] = data_batch_trg['cut_mix_x'
                    ][1][id][del_slice_trg]
                data_batch_trg['cut_mix_x'][0][id] = torch.cat((
                    data_batch_trg['cut_mix_x'][0][id],
                    copy_data_batch_src_cut_mix_x_0[data_batch_src[
                    'cut_sel_indices'][id]]), 0)
                data_batch_trg['cut_mix_x'][1][id] = torch.cat((
                    data_batch_trg['cut_mix_x'][1][id],
                    copy_data_batch_src_cut_mix_x_1[data_batch_src[
                    'cut_sel_indices'][id]]), 0)
            data_batch_src['sam_mix_image'] = data_batch_src['sam_mix_image'
                ].permute(0, 3, 1, 2).cuda()
            data_batch_trg['sam_mix_image'] = data_batch_trg['sam_mix_image'
                ].permute(0, 3, 1, 2).cuda()
            data_batch_src['cut_mix_image'] = data_batch_src['cut_mix_image'
                ].permute(0, 3, 1, 2).cuda()
            data_batch_trg['cut_mix_image'] = data_batch_trg['cut_mix_image'
                ].permute(0, 3, 1, 2).cuda()
            data_batch_src['img'] = data_batch_src['img'].permute(0, 3, 1, 2
                ).cuda()
            data_batch_trg['img'] = data_batch_trg['img'].permute(0, 3, 1, 2
                ).cuda()
            data_batch_src['sam_mix_label_2d'] = torch.cat(data_batch_src[
                'sam_mix_label_2d'], 0).cuda()
            data_batch_src['cut_mix_label_2d'] = torch.cat(data_batch_src[
                'cut_mix_label_2d'], 0).cuda()
            data_batch_src['sam_mix_label_3d'] = torch.cat(data_batch_src[
                'sam_mix_label_3d'], 0).cuda()
            data_batch_src['cut_mix_label_3d'] = torch.cat(data_batch_src[
                'cut_mix_label_3d'], 0).cuda()
            data_batch_trg['sam_mix_pseudo_label_2d'] = torch.cat(
                data_batch_trg['sam_mix_pseudo_label_2d'], 0).cuda()
            data_batch_trg['cut_mix_pseudo_label_2d'] = torch.cat(
                data_batch_trg['cut_mix_pseudo_label_2d'], 0).cuda()
            data_batch_trg['sam_mix_pseudo_label_3d'] = torch.cat(
                data_batch_trg['sam_mix_pseudo_label_3d'], 0).cuda()
            data_batch_trg['cut_mix_pseudo_label_3d'] = torch.cat(
                data_batch_trg['cut_mix_pseudo_label_3d'], 0).cuda()
            data_batch_src['sam_mix_x'][0] = torch.cat(data_batch_src[
                'sam_mix_x'][0], 0)
            data_batch_src['sam_mix_x'][1] = torch.cat(data_batch_src[
                'sam_mix_x'][1], 0)
            data_batch_src['cut_mix_x'][0] = torch.cat(data_batch_src[
                'cut_mix_x'][0], 0)
            data_batch_src['cut_mix_x'][1] = torch.cat(data_batch_src[
                'cut_mix_x'][1], 0)
            data_batch_trg['sam_mix_x'][0] = torch.cat(data_batch_trg[
                'sam_mix_x'][0], 0)
            data_batch_trg['sam_mix_x'][1] = torch.cat(data_batch_trg[
                'sam_mix_x'][1], 0)
            data_batch_trg['cut_mix_x'][0] = torch.cat(data_batch_trg[
                'cut_mix_x'][0], 0)
            data_batch_trg['cut_mix_x'][1] = torch.cat(data_batch_trg[
                'cut_mix_x'][1], 0)
            data_batch_src['sam_mix_x'][1] = data_batch_src['sam_mix_x'][1
                ].cuda()
            data_batch_src['cut_mix_x'][1] = data_batch_src['cut_mix_x'][1
                ].cuda()
            data_batch_trg['sam_mix_x'][1] = data_batch_trg['sam_mix_x'][1
                ].cuda()
            data_batch_trg['cut_mix_x'][1] = data_batch_trg['cut_mix_x'][1
                ].cuda()
            data_batch_src['x'][1] = data_batch_src['x'][1].cuda()
            data_batch_src['seg_label'] = data_batch_src['seg_label'].cuda()
            data_batch_src['img'] = data_batch_src['img'].cuda()
            data_batch_trg['x'][1] = data_batch_trg['x'][1].cuda()
            data_batch_trg['img'] = data_batch_trg['img'].cuda()
            if cfg.TRAIN.XMUDA.lambda_pl > 0:
                data_batch_trg['pseudo_label_2d'] = data_batch_trg[
                    'pseudo_label_2d'].cuda()
                data_batch_trg['pseudo_label_3d'] = data_batch_trg[
                    'pseudo_label_3d'].cuda()
        else:
            raise NotImplementedError('Only SCN is supported for now.')
        data_time = time.time() - end
        if _enable_model_A_:
            optimizer_model_A_2d.zero_grad()
            optimizer_model_A_3d.zero_grad()
        if _enable_model_B_:
            optimizer_model_B_2d.zero_grad()
            optimizer_model_B_3d.zero_grad()
        """
        1. BackProp for Model A
        """
        if _enable_model_A_:
            preds_2d = model_A_2d(data_batch_src)
            preds_3d = model_A_3d(data_batch_src)
            seg_loss_src_2d = F.cross_entropy(preds_2d['seg_logit'],
                data_batch_src['seg_label'], weight=class_weights)
            seg_loss_src_3d = F.cross_entropy(preds_3d['seg_logit'],
                data_batch_src['seg_label'], weight=class_weights)
            train_metric_logger_A.update(seg_loss_src_2d=seg_loss_src_2d,
                seg_loss_src_3d=seg_loss_src_3d)
            loss_2d = seg_loss_src_2d
            loss_3d = seg_loss_src_3d
            if cfg.TRAIN.XMUDA.lambda_xm_src > 0:
                seg_logit_2d = preds_2d['seg_logit2'
                    ] if cfg.MODEL_2D.DUAL_HEAD else preds_2d['seg_logit']
                seg_logit_3d = preds_3d['seg_logit2'
                    ] if cfg.MODEL_3D.DUAL_HEAD else preds_3d['seg_logit']
                xm_loss_src_2d = F.kl_div(F.log_softmax(seg_logit_2d, dim=1
                    ), F.softmax(preds_3d['seg_logit'].detach(), dim=1),
                    reduction='none').sum(1).mean()
                xm_loss_src_3d = F.kl_div(F.log_softmax(seg_logit_3d, dim=1
                    ), F.softmax(preds_2d['seg_logit'].detach(), dim=1),
                    reduction='none').sum(1).mean()
                train_metric_logger_A.update(xm_loss_src_2d=xm_loss_src_2d,
                    xm_loss_src_3d=xm_loss_src_3d)
                loss_2d += cfg.TRAIN.XMUDA.lambda_xm_src * xm_loss_src_2d
                loss_3d += cfg.TRAIN.XMUDA.lambda_xm_src * xm_loss_src_3d
            with torch.no_grad():
                train_metric_model_A_2d.update_dict(preds_2d, data_batch_src)
                train_metric_model_A_3d.update_dict(preds_3d, data_batch_src)
            loss_2d.backward()
            loss_3d.backward()
            preds_2d = model_A_2d(data_batch_trg)
            preds_3d = model_A_3d(data_batch_trg)
            loss_2d = []
            loss_3d = []
            if cfg.TRAIN.XMUDA.lambda_xm_trg > 0:
                seg_logit_2d = preds_2d['seg_logit2'
                    ] if cfg.MODEL_2D.DUAL_HEAD else preds_2d['seg_logit']
                seg_logit_3d = preds_3d['seg_logit2'
                    ] if cfg.MODEL_3D.DUAL_HEAD else preds_3d['seg_logit']
                xm_loss_trg_2d = F.kl_div(F.log_softmax(seg_logit_2d, dim=1
                    ), F.softmax(preds_3d['seg_logit'].detach(), dim=1),
                    reduction='none').sum(1).mean()
                xm_loss_trg_3d = F.kl_div(F.log_softmax(seg_logit_3d, dim=1
                    ), F.softmax(preds_2d['seg_logit'].detach(), dim=1),
                    reduction='none').sum(1).mean()
                train_metric_logger_A.update(xm_loss_trg_2d=xm_loss_trg_2d,
                    xm_loss_trg_3d=xm_loss_trg_3d)
                loss_2d.append(cfg.TRAIN.XMUDA.lambda_xm_trg * xm_loss_trg_2d)
                loss_3d.append(cfg.TRAIN.XMUDA.lambda_xm_trg * xm_loss_trg_3d)
            if cfg.TRAIN.XMUDA.lambda_pl > 0:
                pl_loss_trg_2d = F.cross_entropy(preds_2d['seg_logit'],
                    data_batch_trg['pseudo_label_2d'], weight=class_weights_pl)
                pl_loss_trg_3d = F.cross_entropy(preds_3d['seg_logit'],
                    data_batch_trg['pseudo_label_3d'], weight=class_weights_pl)
                train_metric_logger_A.update(pl_loss_trg_2d=pl_loss_trg_2d,
                    pl_loss_trg_3d=pl_loss_trg_3d)
                loss_2d.append(cfg.TRAIN.XMUDA.lambda_pl * pl_loss_trg_2d)
                loss_3d.append(cfg.TRAIN.XMUDA.lambda_pl * pl_loss_trg_3d)
            if cfg.TRAIN.XMUDA.lambda_minent > 0:
                minent_loss_trg_2d = entropy_loss(F.softmax(preds_2d[
                    'seg_logit'], dim=1))
                minent_loss_trg_3d = entropy_loss(F.softmax(preds_3d[
                    'seg_logit'], dim=1))
                train_metric_logger_A.update(minent_loss_trg_2d=
                    minent_loss_trg_2d, minent_loss_trg_3d=minent_loss_trg_3d)
                loss_2d.append(cfg.TRAIN.XMUDA.lambda_minent *
                    minent_loss_trg_2d)
                loss_3d.append(cfg.TRAIN.XMUDA.lambda_minent *
                    minent_loss_trg_3d)
            sum(loss_2d).backward()
            sum(loss_3d).backward()
            preds_2d = model_A_2d(None, [data_batch_src['sam_mix_image'],
                data_batch_src['sam_mix_indices']])
            preds_3d = model_A_3d(None, data_batch_src['sam_mix_x'])
            seg_loss_src_2d = F.cross_entropy(preds_2d['seg_logit'],
                data_batch_src['sam_mix_label_2d'], weight=class_weights)
            seg_loss_src_3d = F.cross_entropy(preds_3d['seg_logit'],
                data_batch_src['sam_mix_label_3d'], weight=class_weights)
            train_metric_logger_A.update(seg_loss_src_2d=seg_loss_src_2d,
                seg_loss_src_3d=seg_loss_src_3d)
            loss_2d = seg_loss_src_2d
            loss_3d = seg_loss_src_3d
            seg_logit_2d = preds_2d['seg_logit2'
                ] if cfg.MODEL_2D.DUAL_HEAD else preds_2d['seg_logit']
            seg_logit_3d = preds_3d['seg_logit2'
                ] if cfg.MODEL_3D.DUAL_HEAD else preds_3d['seg_logit']
            xm_loss_src_2d = F.kl_div(F.log_softmax(seg_logit_2d, dim=1), F
                .softmax(preds_3d['seg_logit'].detach(), dim=1), reduction=
                'none').sum(1).mean()
            xm_loss_src_3d = F.kl_div(F.log_softmax(seg_logit_3d, dim=1), F
                .softmax(preds_2d['seg_logit'].detach(), dim=1), reduction=
                'none').sum(1).mean()
            train_metric_logger_A.update(xm_loss_src_2d=xm_loss_src_2d,
                xm_loss_src_3d=xm_loss_src_3d)
            loss_2d += cfg_mixed_data_xm_lambda * xm_loss_src_2d
            loss_3d += cfg_mixed_data_xm_lambda * xm_loss_src_3d
            loss_2d.backward()
            loss_3d.backward()
            preds_2d = model_A_2d(None, [data_batch_trg['sam_mix_image'],
                data_batch_trg['sam_mix_indices']])
            preds_3d = model_A_3d(None, data_batch_trg['sam_mix_x'])
            loss_2d = []
            loss_3d = []
            pl_loss_trg_2d = F.cross_entropy(preds_2d['seg_logit'],
                data_batch_trg['sam_mix_pseudo_label_2d'], weight=
                class_weights_pl)
            pl_loss_trg_3d = F.cross_entropy(preds_3d['seg_logit'],
                data_batch_trg['sam_mix_pseudo_label_3d'], weight=
                class_weights_pl)
            train_metric_logger_A.update(pl_loss_trg_2d=pl_loss_trg_2d,
                pl_loss_trg_3d=pl_loss_trg_3d)
            loss_2d.append(pl_loss_trg_2d)
            loss_3d.append(pl_loss_trg_3d)
            seg_logit_2d = preds_2d['seg_logit2'
                ] if cfg.MODEL_2D.DUAL_HEAD else preds_2d['seg_logit']
            seg_logit_3d = preds_3d['seg_logit2'
                ] if cfg.MODEL_3D.DUAL_HEAD else preds_3d['seg_logit']
            xm_loss_trg_2d = F.kl_div(F.log_softmax(seg_logit_2d, dim=1), F
                .softmax(preds_3d['seg_logit'].detach(), dim=1), reduction=
                'none').sum(1).mean()
            xm_loss_trg_3d = F.kl_div(F.log_softmax(seg_logit_3d, dim=1), F
                .softmax(preds_2d['seg_logit'].detach(), dim=1), reduction=
                'none').sum(1).mean()
            train_metric_logger_A.update(xm_loss_trg_2d=xm_loss_trg_2d,
                xm_loss_trg_3d=xm_loss_trg_3d)
            loss_2d.append(cfg_mixed_data_xm_lambda * xm_loss_trg_2d)
            loss_3d.append(cfg_mixed_data_xm_lambda * xm_loss_trg_3d)
            sum(loss_2d).backward()
            sum(loss_3d).backward()
            optimizer_model_A_2d.step()
            optimizer_model_A_3d.step()
            scheduler_model_A_2d.step()
            scheduler_model_A_3d.step()
        """
        2. BackProp for Model B
        """
        if _enable_model_B_:
            preds_2d = model_B_2d(data_batch_src)
            preds_3d = model_B_3d(data_batch_src)
            seg_loss_src_2d = F.cross_entropy(preds_2d['seg_logit'],
                data_batch_src['seg_label'], weight=class_weights)
            seg_loss_src_3d = F.cross_entropy(preds_3d['seg_logit'],
                data_batch_src['seg_label'], weight=class_weights)
            loss_2d = seg_loss_src_2d
            loss_3d = seg_loss_src_3d
            if cfg.TRAIN.XMUDA.lambda_xm_src > 0:
                seg_logit_2d = preds_2d['seg_logit2'
                    ] if cfg.MODEL_2D.DUAL_HEAD else preds_2d['seg_logit']
                seg_logit_3d = preds_3d['seg_logit2'
                    ] if cfg.MODEL_3D.DUAL_HEAD else preds_3d['seg_logit']
                xm_loss_src_2d = F.kl_div(F.log_softmax(seg_logit_2d, dim=1
                    ), F.softmax(preds_3d['seg_logit'].detach(), dim=1),
                    reduction='none').sum(1).mean()
                xm_loss_src_3d = F.kl_div(F.log_softmax(seg_logit_3d, dim=1
                    ), F.softmax(preds_2d['seg_logit'].detach(), dim=1),
                    reduction='none').sum(1).mean()
                loss_2d += cfg.TRAIN.XMUDA.lambda_xm_src * xm_loss_src_2d
                loss_3d += cfg.TRAIN.XMUDA.lambda_xm_src * xm_loss_src_3d
            with torch.no_grad():
                train_metric_model_B_2d.update_dict(preds_2d, data_batch_src)
                train_metric_model_B_3d.update_dict(preds_3d, data_batch_src)
            loss_2d.backward()
            loss_3d.backward()
            preds_2d = model_B_2d(data_batch_trg)
            preds_3d = model_B_3d(data_batch_trg)
            loss_2d = []
            loss_3d = []
            if cfg.TRAIN.XMUDA.lambda_xm_trg > 0:
                seg_logit_2d = preds_2d['seg_logit2'
                    ] if cfg.MODEL_2D.DUAL_HEAD else preds_2d['seg_logit']
                seg_logit_3d = preds_3d['seg_logit2'
                    ] if cfg.MODEL_3D.DUAL_HEAD else preds_3d['seg_logit']
                xm_loss_trg_2d = F.kl_div(F.log_softmax(seg_logit_2d, dim=1
                    ), F.softmax(preds_3d['seg_logit'].detach(), dim=1),
                    reduction='none').sum(1).mean()
                xm_loss_trg_3d = F.kl_div(F.log_softmax(seg_logit_3d, dim=1
                    ), F.softmax(preds_2d['seg_logit'].detach(), dim=1),
                    reduction='none').sum(1).mean()
                train_metric_logger_A.update(xm_loss_trg_2d=xm_loss_trg_2d,
                    xm_loss_trg_3d=xm_loss_trg_3d)
                loss_2d.append(cfg.TRAIN.XMUDA.lambda_xm_trg * xm_loss_trg_2d)
                loss_3d.append(cfg.TRAIN.XMUDA.lambda_xm_trg * xm_loss_trg_3d)
            if cfg.TRAIN.XMUDA.lambda_pl > 0:
                pl_loss_trg_2d = F.cross_entropy(preds_2d['seg_logit'],
                    data_batch_trg['pseudo_label_2d'], weight=class_weights_pl)
                pl_loss_trg_3d = F.cross_entropy(preds_3d['seg_logit'],
                    data_batch_trg['pseudo_label_3d'], weight=class_weights_pl)
                train_metric_logger_A.update(pl_loss_trg_2d=pl_loss_trg_2d,
                    pl_loss_trg_3d=pl_loss_trg_3d)
                loss_2d.append(cfg.TRAIN.XMUDA.lambda_pl * pl_loss_trg_2d)
                loss_3d.append(cfg.TRAIN.XMUDA.lambda_pl * pl_loss_trg_3d)
            if cfg.TRAIN.XMUDA.lambda_minent > 0:
                minent_loss_trg_2d = entropy_loss(F.softmax(preds_2d[
                    'seg_logit'], dim=1))
                minent_loss_trg_3d = entropy_loss(F.softmax(preds_3d[
                    'seg_logit'], dim=1))
                train_metric_logger_A.update(minent_loss_trg_2d=
                    minent_loss_trg_2d, minent_loss_trg_3d=minent_loss_trg_3d)
                loss_2d.append(cfg.TRAIN.XMUDA.lambda_minent *
                    minent_loss_trg_2d)
                loss_3d.append(cfg.TRAIN.XMUDA.lambda_minent *
                    minent_loss_trg_3d)
            sum(loss_2d).backward()
            sum(loss_3d).backward()
            preds_2d = model_B_2d(None, [data_batch_src['sam_mix_image'],
                data_batch_src['sam_mix_indices']])
            preds_3d = model_B_3d(None, data_batch_src['sam_mix_x'])
            seg_loss_src_2d = F.cross_entropy(preds_2d['seg_logit'],
                data_batch_src['sam_mix_label_2d'], weight=class_weights)
            seg_loss_src_3d = F.cross_entropy(preds_3d['seg_logit'],
                data_batch_src['sam_mix_label_3d'], weight=class_weights)
            loss_2d = seg_loss_src_2d
            loss_3d = seg_loss_src_3d
            seg_logit_2d = preds_2d['seg_logit2'
                ] if cfg.MODEL_2D.DUAL_HEAD else preds_2d['seg_logit']
            seg_logit_3d = preds_3d['seg_logit2'
                ] if cfg.MODEL_3D.DUAL_HEAD else preds_3d['seg_logit']
            xm_loss_src_2d = F.kl_div(F.log_softmax(seg_logit_2d, dim=1), F
                .softmax(preds_3d['seg_logit'].detach(), dim=1), reduction=
                'none').sum(1).mean()
            xm_loss_src_3d = F.kl_div(F.log_softmax(seg_logit_3d, dim=1), F
                .softmax(preds_2d['seg_logit'].detach(), dim=1), reduction=
                'none').sum(1).mean()
            train_metric_logger_A.update(xm_loss_src_2d=xm_loss_src_2d,
                xm_loss_src_3d=xm_loss_src_3d)
            loss_2d += cfg_mixed_data_xm_lambda * xm_loss_src_2d
            loss_3d += cfg_mixed_data_xm_lambda * xm_loss_src_3d
            loss_2d.backward()
            loss_3d.backward()
            preds_2d = model_B_2d(None, [data_batch_trg['sam_mix_image'],
                data_batch_trg['sam_mix_indices']])
            preds_3d = model_B_3d(None, data_batch_trg['sam_mix_x'])
            loss_2d = []
            loss_3d = []
            pl_loss_trg_2d = F.cross_entropy(preds_2d['seg_logit'],
                data_batch_trg['sam_mix_pseudo_label_2d'], weight=
                class_weights_pl)
            pl_loss_trg_3d = F.cross_entropy(preds_3d['seg_logit'],
                data_batch_trg['sam_mix_pseudo_label_3d'], weight=
                class_weights_pl)
            loss_2d.append(pl_loss_trg_2d)
            loss_3d.append(pl_loss_trg_3d)
            seg_logit_2d = preds_2d['seg_logit2'
                ] if cfg.MODEL_2D.DUAL_HEAD else preds_2d['seg_logit']
            seg_logit_3d = preds_3d['seg_logit2'
                ] if cfg.MODEL_3D.DUAL_HEAD else preds_3d['seg_logit']
            xm_loss_trg_2d = F.kl_div(F.log_softmax(seg_logit_2d, dim=1), F
                .softmax(preds_3d['seg_logit'].detach(), dim=1), reduction=
                'none').sum(1).mean()
            xm_loss_trg_3d = F.kl_div(F.log_softmax(seg_logit_3d, dim=1), F
                .softmax(preds_2d['seg_logit'].detach(), dim=1), reduction=
                'none').sum(1).mean()
            loss_2d.append(cfg_mixed_data_xm_lambda * xm_loss_trg_2d)
            loss_3d.append(cfg_mixed_data_xm_lambda * xm_loss_trg_3d)
            sum(loss_2d).backward()
            sum(loss_3d).backward()
            optimizer_model_B_2d.step()
            optimizer_model_B_3d.step()
            scheduler_model_B_2d.step()
            scheduler_model_B_3d.step()
        batch_time = time.time() - end
        if _enable_model_A_:
            train_metric_logger_A.update(time=batch_time, data=data_time)
        if _enable_model_B_:
            train_metric_logger_B.update(time=batch_time, data=data_time)
        cur_iter = iteration + 1
        if (cfg.TRAIN.LOG_PERIOD > 0 and cur_iter % cfg.TRAIN.LOG_PERIOD ==
            0 or cur_iter == 1):
            if _enable_model_A_:
                logger.info(train_metric_logger_A.delimiter.join([
                    'iter: {iter:4d}', '{meters}', 'lr: {lr:.2e}',
                    'max mem: {memory:.0f}']).format(iter=cur_iter, meters=
                    str(train_metric_logger_A), lr=optimizer_model_A_2d.
                    param_groups[0]['lr'], memory=torch.cuda.
                    max_memory_allocated() / 1024.0 ** 2))
            if _enable_model_B_:
                logger.info(train_metric_logger_B.delimiter.join([
                    'iter: {iter:4d}', '{meters}', 'lr: {lr:.2e}',
                    'max mem: {memory:.0f}']).format(iter=cur_iter, meters=
                    str(train_metric_logger_B), lr=optimizer_model_B_2d.
                    param_groups[0]['lr'], memory=torch.cuda.
                    max_memory_allocated() / 1024.0 ** 2))
        if (ckpt_period > 0 and cur_iter % ckpt_period == 0 or cur_iter ==
            max_iteration):
            if _enable_model_A_:
                checkpoint_data_model_A_2d['iteration'] = cur_iter
                checkpoint_data_model_A_2d[best_metric_name] = best_metric['2d'
                    ]
                checkpointer_model_A_2d.save('model_A_2d_{:06d}'.format(
                    cur_iter), **checkpoint_data_model_A_2d)
                checkpoint_data_model_A_3d['iteration'] = cur_iter
                checkpoint_data_model_A_3d[best_metric_name] = best_metric['3d'
                    ]
                checkpointer_model_A_3d.save('model_A_3d_{:06d}'.format(
                    cur_iter), **checkpoint_data_model_A_3d)
            if _enable_model_B_:
                checkpoint_data_model_B_2d['iteration'] = cur_iter
                checkpoint_data_model_B_2d[best_metric_name] = best_metric['2d'
                    ]
                checkpointer_model_B_2d.save('model_B_2d_{:06d}'.format(
                    cur_iter), **checkpoint_data_model_B_2d)
                checkpoint_data_model_B_3d['iteration'] = cur_iter
                checkpoint_data_model_B_3d[best_metric_name] = best_metric['3d'
                    ]
                checkpointer_model_B_3d.save('model_B_3d_{:06d}'.format(
                    cur_iter), **checkpoint_data_model_B_3d)
        gc.collect()
        torch.cuda.empty_cache()
        end = time.time()
