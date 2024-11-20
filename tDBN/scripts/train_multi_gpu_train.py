def train(config_path, model_dir, result_path=None, create_folder=False,
    display_step=50, summary_step=5, device_ids=[0, 1], pickle_result=True):
    """train a VoxelNet model specified by a config file.
    """
    if create_folder:
        if pathlib.Path(model_dir).exists():
            model_dir = torchplus.train.create_folder(model_dir)
    model_dir = pathlib.Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    if result_path is None:
        result_path = model_dir / 'results'
    config_file_bkp = 'pipeline.config'
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, 'r') as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    shutil.copyfile(config_path, str(model_dir / config_file_bkp))
    input_cfg = config.train_input_reader
    eval_input_cfg = config.eval_input_reader
    model_cfg = config.model.tDBN
    train_cfg = config.train_config
    class_names = list(input_cfg.class_names)
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
        bv_range, box_coder)
    center_limit_range = model_cfg.post_center_limit_range
    net = tDBN_builder.build(model_cfg, voxel_generator, target_assigner)
    torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    net_paral = multi_gpu_parallel(net, device_ids=[0, 1]).cuda()
    print('num_trainable parameters:', len(list(net_paral.module.parameters()))
        )
    gstep = net_paral.module.get_global_step() - 1
    optimizer_cfg = train_cfg.optimizer
    optimizer = optimizer_builder.build(optimizer_cfg, net_paral.module.
        parameters())
    torchplus.train.try_restore_latest_checkpoints(model_dir, [optimizer])
    optimizer = torch.nn.DataParallel(optimizer, device_ids=device_ids)
    lr_scheduler = lr_scheduler_builder.build(optimizer_cfg, optimizer.
        module, gstep)
    float_dtype = torch.float32
    dataset = input_reader_builder.build(input_cfg, model_cfg, training=
        True, voxel_generator=voxel_generator, target_assigner=target_assigner)
    eval_dataset = input_reader_builder.build(eval_input_cfg, model_cfg,
        training=False, voxel_generator=voxel_generator, target_assigner=
        target_assigner)

    def _worker_init_fn(worker_id):
        time_seed = np.array(time.time(), dtype=np.int32)
        np.random.seed(time_seed + worker_id)
        print(f'WORKER {worker_id} seed:', np.random.get_state()[1][0])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=input_cfg.
        batch_size, shuffle=True, num_workers=input_cfg.num_workers,
        pin_memory=False, collate_fn=merge_tDBN_batch_training_pass,
        worker_init_fn=_worker_init_fn)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=
        eval_input_cfg.batch_size, shuffle=False, num_workers=
        eval_input_cfg.num_workers, pin_memory=False, collate_fn=
        merge_tDBN_batch)
    data_iter = iter(dataloader)
    log_path = model_dir / 'log.txt'
    eval_log_path = model_dir / 'eval_log.txt'
    logf = open(log_path, 'a')
    evallogf = open(eval_log_path, 'a')
    logf.write(proto_str)
    logf.write('\n')
    summary_dir = model_dir / 'summary'
    summary_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(summary_dir))
    total_step_elapsed = 0
    remain_steps = train_cfg.steps - net_paral.module.get_global_step()
    t = time.time()
    ckpt_start_time = t
    total_loop = train_cfg.steps // train_cfg.steps_per_eval + 1
    clear_metrics_every_epoch = train_cfg.clear_metrics_every_epoch
    if train_cfg.steps % train_cfg.steps_per_eval == 0:
        total_loop -= 1
    optimizer.module.zero_grad()
    try:
        for _ in range(total_loop):
            if total_step_elapsed + train_cfg.steps_per_eval > train_cfg.steps:
                steps = train_cfg.steps % train_cfg.steps_per_eval
            else:
                steps = train_cfg.steps_per_eval
            for step in range(steps):
                lr_scheduler.step()
                try:
                    example = next(data_iter)
                except StopIteration:
                    print('end epoch')
                    if clear_metrics_every_epoch:
                        net_paral.module.clear_metrics()
                    data_iter = iter(dataloader)
                    example = next(data_iter)
                batch_size = len(example)
                ret_dict = net_paral(example)
                cls_preds = ret_dict['cls_preds']
                loss = ret_dict['loss'].mean()
                cls_loss_reduced = ret_dict['cls_loss_reduced'].mean()
                loc_loss_reduced = ret_dict['loc_loss_reduced'].mean()
                cared = ret_dict['cared']
                if train_cfg.enable_mixed_precision:
                    loss *= loss_scale
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net_paral.module.parameters(
                    ), 10.0)
                optimizer.module.step()
                optimizer.module.zero_grad()
                net_paral.module.update_global_step()
                temp = []
                for v in example:
                    temp.append(v['labels'])
                labels = np.stack(temp, axis=0)
                labels = torch.as_tensor(labels, dtype=torch.int32, device=0)
                net_metrics = net_paral.module.update_metrics(cls_loss_reduced,
                    loc_loss_reduced, cls_preds, labels, cared)
                step_time = time.time() - t
                t = time.time()
                metrics = {}
                global_step = net_paral.module.get_global_step()
                if global_step % display_step == 0:
                    cls_pos_loss = ret_dict['cls_pos_loss'].mean()
                    cls_neg_loss = ret_dict['cls_neg_loss'].mean()
                    dir_loss_reduced = ret_dict['dir_loss_reduced'].mean()
                    loc_loss = ret_dict['loc_loss']
                    cls_loss = ret_dict['cls_loss']
                    num_pos = int((labels > 0)[0].float().sum().cpu().numpy())
                    num_neg = int((labels == 0)[0].float().sum().cpu().numpy())
                    loc_loss_elem = [float(loc_loss[:, :, i].sum().detach()
                        .cpu().numpy() / batch_size) for i in range(
                        loc_loss.shape[-1])]
                    metrics['step'] = global_step
                    metrics['steptime'] = step_time
                    metrics.update(net_metrics)
                    metrics['loss'] = {}
                    metrics['loss']['loc_elem'] = loc_loss_elem
                    metrics['loss']['cls_pos_rt'] = float(cls_pos_loss.
                        detach().cpu().numpy())
                    metrics['loss']['cls_neg_rt'] = float(cls_neg_loss.
                        detach().cpu().numpy())
                    if model_cfg.use_direction_classifier:
                        metrics['loss']['dir_rt'] = float(dir_loss_reduced.
                            detach().cpu().numpy())
                    metrics['num_pos'] = int(num_pos)
                    metrics['num_neg'] = int(num_neg)
                    metrics['lr'] = float(optimizer.module.param_groups[0][
                        'lr'])
                    metrics['image_idx'] = example[0]['image_idx']
                    flatted_metrics = flat_nested_json_dict(metrics)
                    flatted_summarys = flat_nested_json_dict(metrics, '/')
                    for k, v in flatted_summarys.items():
                        if isinstance(v, (list, tuple)):
                            v = {str(i): e for i, e in enumerate(v)}
                            writer.add_scalars(k, v, global_step)
                        else:
                            writer.add_scalar(k, v, global_step)
                    metrics_str_list = []
                    for k, v in flatted_metrics.items():
                        if isinstance(v, float):
                            metrics_str_list.append(f'{k}={v:.3}')
                        elif isinstance(v, (list, tuple)):
                            if v and isinstance(v[0], float):
                                v_str = ', '.join([f'{e:.3}' for e in v])
                                metrics_str_list.append(f'{k}=[{v_str}]')
                            else:
                                metrics_str_list.append(f'{k}={v}')
                        else:
                            metrics_str_list.append(f'{k}={v}')
                    log_str = ', '.join(metrics_str_list)
                    print(log_str, file=logf)
                    print(log_str)
                ckpt_elasped_time = time.time() - ckpt_start_time
                if ckpt_elasped_time > train_cfg.save_checkpoints_secs:
                    torchplus.train.save_models(model_dir, [net_paral.
                        module, optimizer.module], net_paral.module.
                        get_global_step())
                    ckpt_start_time = time.time()
            total_step_elapsed += steps
            torchplus.train.save_models(model_dir, [net_paral.module,
                optimizer.module], net_paral.module.get_global_step())
            net_paral.module.eval()
            result_path_step = result_path / f'step_{net.get_global_step()}'
            result_path_step.mkdir(parents=True, exist_ok=True)
            print('#################################')
            print('#################################', file=logf)
            print('# EVAL')
            print('# EVAL', file=logf)
            print('Eval_at_{}'.format(net.get_global_step()), file=evallogf)
            print('#################################')
            print('#################################', file=logf)
            print('Generate output labels...')
            print('Generate output labels...', file=logf)
            t = time.time()
            dt_annos = []
            prog_bar = ProgressBar()
            prog_bar.start(len(eval_dataset) // eval_input_cfg.batch_size + 1)
            for example in iter(eval_dataloader):
                example = example_convert_to_torch(example, float_dtype)
                if pickle_result:
                    dt_annos += predict_kitti_to_anno(net_paral.module,
                        example, class_names, center_limit_range, model_cfg
                        .lidar_input)
                else:
                    _predict_kitti_to_file(net_paral.module, example,
                        result_path_step, class_names, center_limit_range,
                        model_cfg.lidar_input)
                prog_bar.print_bar()
            sec_per_ex = len(eval_dataset) / (time.time() - t)
            print(
                f'avg forward time per example: {net_paral.module.avg_forward_time:.3f}'
                )
            print(
                f'avg postprocess time per example: {net_paral.module.avg_postprocess_time:.3f}'
                )
            net_paral.module.clear_time_metrics()
            print(f'generate label finished({sec_per_ex:.2f}/s). start eval:')
            print(f'generate label finished({sec_per_ex:.2f}/s). start eval:',
                file=logf)
            gt_annos = [info['annos'] for info in eval_dataset.dataset.
                kitti_infos]
            if not pickle_result:
                dt_annos = kitti.get_label_annos(result_path_step)
            result = get_official_eval_result(gt_annos, dt_annos, class_names)
            print(result, file=logf)
            print(result, file=evallogf)
            print(result)
            writer.add_text('eval_result', result, global_step)
            result = get_coco_eval_result(gt_annos, dt_annos, class_names)
            print(result, file=logf)
            print(result, file=evallogf)
            print(result)
            if pickle_result:
                with open(result_path_step / 'result.pkl', 'wb') as f:
                    pickle.dump(dt_annos, f)
            writer.add_text('eval_result', result, global_step)
            net_paral.module.train()
    except Exception as e:
        torchplus.train.save_models(model_dir, [net_paral.module, optimizer
            .module], net_paral.module.get_global_step())
        logf.close()
        evallogf.close()
        raise e
    torchplus.train.save_models(model_dir, [net_paral.module, optimizer.
        module], net_paral.module.get_global_step())
    logf.close()
    evallogf.close()
