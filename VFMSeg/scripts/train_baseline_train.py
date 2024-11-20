def train(cfg, output_dir='', run_name=''):
    logger = logging.getLogger('xmuda.train')
    set_random_seed(cfg.RNG_SEED)
    model_2d, train_metric_2d = build_model_2d(cfg)
    logger.info('Build 2D model:\n{}'.format(str(model_2d)))
    num_params = sum(param.numel() for param in model_2d.parameters())
    print('#Parameters: {:.2e}'.format(num_params))
    model_3d, train_metric_3d = build_model_3d(cfg)
    logger.info('Build 3D model:\n{}'.format(str(model_3d)))
    num_params = sum(param.numel() for param in model_3d.parameters())
    print('#Parameters: {:.2e}'.format(num_params))
    model_2d = model_2d.cuda()
    model_3d = model_3d.cuda()
    optimizer_2d = build_optimizer(cfg, model_2d)
    optimizer_3d = build_optimizer(cfg, model_3d)
    scheduler_2d = build_scheduler(cfg, optimizer_2d)
    scheduler_3d = build_scheduler(cfg, optimizer_3d)
    checkpointer_2d = CheckpointerV2(model_2d, optimizer=optimizer_2d,
        scheduler=scheduler_2d, save_dir=output_dir, logger=logger, postfix
        ='_2d', max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    checkpoint_data_2d = checkpointer_2d.load(cfg.RESUME_PATH, resume=cfg.
        AUTO_RESUME, resume_states=cfg.RESUME_STATES)
    checkpointer_3d = CheckpointerV2(model_3d, optimizer=optimizer_3d,
        scheduler=scheduler_3d, save_dir=output_dir, logger=logger, postfix
        ='_3d', max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    checkpoint_data_3d = checkpointer_3d.load(cfg.RESUME_PATH, resume=cfg.
        AUTO_RESUME, resume_states=cfg.RESUME_STATES)
    ckpt_period = cfg.TRAIN.CHECKPOINT_PERIOD
    if output_dir:
        tb_dir = osp.join(output_dir, 'tb.{:s}'.format(run_name))
        summary_writer = SummaryWriter(tb_dir)
    else:
        summary_writer = None
    max_iteration = cfg.SCHEDULER.MAX_ITERATION
    start_iteration = checkpoint_data_2d.get('iteration', 0)
    set_random_seed(cfg.RNG_SEED)
    train_dataloader_src = build_dataloader(cfg, mode='train', domain=
        'source', start_iteration=start_iteration)
    val_period = cfg.VAL.PERIOD
    val_dataloader = build_dataloader(cfg, mode='val', domain='target'
        ) if val_period > 0 else None
    best_metric_name = 'best_{}'.format(cfg.VAL.METRIC)
    best_metric = {'2d': checkpoint_data_2d.get(best_metric_name, None),
        '3d': checkpoint_data_3d.get(best_metric_name, None)}
    best_metric_iter = {'2d': -1, '3d': -1}
    logger.info('Start training from iteration {}'.format(start_iteration))
    train_metric_logger = init_metric_logger([train_metric_2d, train_metric_3d]
        )
    val_metric_logger = MetricLogger(delimiter='  ')

    def setup_train():
        model_2d.train()
        model_3d.train()
        train_metric_logger.reset()

    def setup_validate():
        model_2d.eval()
        model_3d.eval()
        val_metric_logger.reset()
    if cfg.TRAIN.CLASS_WEIGHTS:
        class_weights = torch.tensor(cfg.TRAIN.CLASS_WEIGHTS).cuda()
    else:
        class_weights = None
    setup_train()
    end = time.time()
    train_iter_src = enumerate(train_dataloader_src)
    for iteration in range(start_iteration, max_iteration):
        _, data_batch_src = train_iter_src.__next__()
        data_time = time.time() - end
        if ('SCN' in cfg.DATASET_SOURCE.TYPE and 'SCN' in cfg.
            DATASET_TARGET.TYPE):
            data_batch_src['x'][1] = data_batch_src['x'][1].cuda()
            data_batch_src['seg_label'] = data_batch_src['seg_label'].cuda()
            data_batch_src['img'] = data_batch_src['img'].cuda()
        else:
            raise NotImplementedError('Only SCN is supported for now.')
        optimizer_2d.zero_grad()
        optimizer_3d.zero_grad()
        preds_2d = model_2d(data_batch_src)
        preds_3d = model_3d(data_batch_src)
        seg_loss_src_2d = F.cross_entropy(preds_2d['seg_logit'],
            data_batch_src['seg_label'], weight=class_weights)
        seg_loss_src_3d = F.cross_entropy(preds_3d['seg_logit'],
            data_batch_src['seg_label'], weight=class_weights)
        train_metric_logger.update(seg_loss_src_2d=seg_loss_src_2d,
            seg_loss_src_3d=seg_loss_src_3d)
        loss_2d = seg_loss_src_2d
        loss_3d = seg_loss_src_3d
        if cfg.TRAIN.XMUDA.lambda_xm_src > 0:
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
            train_metric_logger.update(xm_loss_src_2d=xm_loss_src_2d,
                xm_loss_src_3d=xm_loss_src_3d)
            loss_2d += cfg.TRAIN.XMUDA.lambda_xm_src * xm_loss_src_2d
            loss_3d += cfg.TRAIN.XMUDA.lambda_xm_src * xm_loss_src_3d
        with torch.no_grad():
            train_metric_2d.update_dict(preds_2d, data_batch_src)
            train_metric_3d.update_dict(preds_3d, data_batch_src)
        loss_2d.backward()
        loss_3d.backward()
        optimizer_2d.step()
        optimizer_3d.step()
        batch_time = time.time() - end
        train_metric_logger.update(time=batch_time, data=data_time)
        cur_iter = iteration + 1
        if (cur_iter == 1 or cfg.TRAIN.LOG_PERIOD > 0 and cur_iter % cfg.
            TRAIN.LOG_PERIOD == 0):
            logger.info(train_metric_logger.delimiter.join([
                'iter: {iter:4d}', '{meters}', 'lr: {lr:.2e}',
                'max mem: {memory:.0f}']).format(iter=cur_iter, meters=str(
                train_metric_logger), lr=optimizer_2d.param_groups[0]['lr'],
                memory=torch.cuda.max_memory_allocated() / 1024.0 ** 2))
        if (summary_writer is not None and cfg.TRAIN.SUMMARY_PERIOD > 0 and
            cur_iter % cfg.TRAIN.SUMMARY_PERIOD == 0):
            keywords = 'loss', 'acc', 'iou'
            for name, meter in train_metric_logger.meters.items():
                if all(k not in name for k in keywords):
                    continue
                summary_writer.add_scalar('train/' + name, meter.avg,
                    global_step=cur_iter)
        if (ckpt_period > 0 and cur_iter % ckpt_period == 0 or cur_iter ==
            max_iteration):
            checkpoint_data_2d['iteration'] = cur_iter
            checkpoint_data_2d[best_metric_name] = best_metric['2d']
            checkpointer_2d.save('model_2d_{:06d}'.format(cur_iter), **
                checkpoint_data_2d)
            checkpoint_data_3d['iteration'] = cur_iter
            checkpoint_data_3d[best_metric_name] = best_metric['3d']
            checkpointer_3d.save('model_3d_{:06d}'.format(cur_iter), **
                checkpoint_data_3d)
        if val_period > 0 and (cur_iter % val_period == 0 or cur_iter ==
            max_iteration):
            start_time_val = time.time()
            setup_validate()
            validate(cfg, model_2d, model_3d, val_dataloader, val_metric_logger
                )
            epoch_time_val = time.time() - start_time_val
            logger.info('Iteration[{}]-Val {}  total_time: {:.2f}s'.format(
                cur_iter, val_metric_logger.summary_str, epoch_time_val))
            if summary_writer is not None:
                keywords = 'loss', 'acc', 'iou'
                for name, meter in val_metric_logger.meters.items():
                    if all(k not in name for k in keywords):
                        continue
                    summary_writer.add_scalar('val/' + name, meter.avg,
                        global_step=cur_iter)
            for modality in ['2d', '3d']:
                cur_metric_name = cfg.VAL.METRIC + '_' + modality
                if cur_metric_name in val_metric_logger.meters:
                    cur_metric = val_metric_logger.meters[cur_metric_name
                        ].global_avg
                    if best_metric[modality] is None or best_metric[modality
                        ] < cur_metric:
                        best_metric[modality] = cur_metric
                        best_metric_iter[modality] = cur_iter
            setup_train()
        scheduler_2d.step()
        scheduler_3d.step()
        end = time.time()
    for modality in ['2d', '3d']:
        logger.info('Best val-{}-{} = {:.2f} at iteration {}'.format(
            modality.upper(), cfg.VAL.METRIC, best_metric[modality] * 100,
            best_metric_iter[modality]))
