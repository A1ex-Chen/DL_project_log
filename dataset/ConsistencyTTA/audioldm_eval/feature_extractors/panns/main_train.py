def train(args):
    """Train AudioSet tagging model.

    Args:
      dataset_dir: str
      workspace: str
      data_type: 'balanced_train' | 'full_train'
      window_size: int
      hop_size: int
      mel_bins: int
      model_type: str
      loss_type: 'clip_bce'
      balanced: 'none' | 'balanced' | 'alternate'
      augmentation: 'none' | 'mixup'
      batch_size: int
      learning_rate: float
      resume_iteration: int
      early_stop: int
      accumulation_steps: int
      cuda: bool
    """
    workspace = args.workspace
    data_type = args.data_type
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    loss_type = args.loss_type
    balanced = args.balanced
    augmentation = args.augmentation
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    resume_iteration = args.resume_iteration
    early_stop = args.early_stop
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available(
        ) else torch.device('cpu')
    filename = args.filename
    num_workers = 8
    clip_samples = config.clip_samples
    classes_num = config.classes_num
    loss_func = get_loss_func(loss_type)
    black_list_csv = None
    train_indexes_hdf5_path = os.path.join(workspace, 'hdf5s', 'indexes',
        '{}.h5'.format(data_type))
    eval_bal_indexes_hdf5_path = os.path.join(workspace, 'hdf5s', 'indexes',
        'balanced_train.h5')
    eval_test_indexes_hdf5_path = os.path.join(workspace, 'hdf5s',
        'indexes', 'eval.h5')
    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename,
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'
        .format(sample_rate, window_size, hop_size, mel_bins, fmin, fmax),
        'data_type={}'.format(data_type), model_type, 'loss_type={}'.format
        (loss_type), 'balanced={}'.format(balanced), 'augmentation={}'.
        format(augmentation), 'batch_size={}'.format(batch_size))
    create_folder(checkpoints_dir)
    statistics_path = os.path.join(workspace, 'statistics', filename,
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'
        .format(sample_rate, window_size, hop_size, mel_bins, fmin, fmax),
        'data_type={}'.format(data_type), model_type, 'loss_type={}'.format
        (loss_type), 'balanced={}'.format(balanced), 'augmentation={}'.
        format(augmentation), 'batch_size={}'.format(batch_size),
        'statistics.pkl')
    create_folder(os.path.dirname(statistics_path))
    logs_dir = os.path.join(workspace, 'logs', filename,
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'
        .format(sample_rate, window_size, hop_size, mel_bins, fmin, fmax),
        'data_type={}'.format(data_type), model_type, 'loss_type={}'.format
        (loss_type), 'balanced={}'.format(balanced), 'augmentation={}'.
        format(augmentation), 'batch_size={}'.format(batch_size))
    create_logging(logs_dir, filemode='w')
    logging.info(args)
    if 'cuda' in str(device):
        logging.info('Using GPU.')
        device = 'cuda'
    else:
        logging.info('Using CPU. Set --cuda flag to use GPU.')
        device = 'cpu'
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size,
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax,
        classes_num=classes_num)
    params_num = count_parameters(model)
    logging.info('Parameters num: {}'.format(params_num))
    dataset = AudioSetDataset(sample_rate=sample_rate)
    if balanced == 'none':
        Sampler = TrainSampler
    elif balanced == 'balanced':
        Sampler = BalancedTrainSampler
    elif balanced == 'alternate':
        Sampler = AlternateTrainSampler
    train_sampler = Sampler(indexes_hdf5_path=train_indexes_hdf5_path,
        batch_size=batch_size * 2 if 'mixup' in augmentation else
        batch_size, black_list_csv=black_list_csv)
    eval_bal_sampler = EvaluateSampler(indexes_hdf5_path=
        eval_bal_indexes_hdf5_path, batch_size=batch_size)
    eval_test_sampler = EvaluateSampler(indexes_hdf5_path=
        eval_test_indexes_hdf5_path, batch_size=batch_size)
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
        batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=
        num_workers, pin_memory=True)
    eval_bal_loader = torch.utils.data.DataLoader(dataset=dataset,
        batch_sampler=eval_bal_sampler, collate_fn=collate_fn, num_workers=
        num_workers, pin_memory=True)
    eval_test_loader = torch.utils.data.DataLoader(dataset=dataset,
        batch_sampler=eval_test_sampler, collate_fn=collate_fn, num_workers
        =num_workers, pin_memory=True)
    if 'mixup' in augmentation:
        mixup_augmenter = Mixup(mixup_alpha=1.0)
    evaluator = Evaluator(model=model)
    statistics_container = StatisticsContainer(statistics_path)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9,
        0.999), eps=1e-08, weight_decay=0.0, amsgrad=True)
    train_bgn_time = time.time()
    if resume_iteration > 0:
        resume_checkpoint_path = os.path.join(workspace, 'checkpoints',
            filename,
            'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'
            .format(sample_rate, window_size, hop_size, mel_bins, fmin,
            fmax), 'data_type={}'.format(data_type), model_type,
            'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced
            ), 'augmentation={}'.format(augmentation), 'batch_size={}'.
            format(batch_size), '{}_iterations.pth'.format(resume_iteration))
        logging.info('Loading checkpoint {}'.format(resume_checkpoint_path))
        checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        train_sampler.load_state_dict(checkpoint['sampler'])
        statistics_container.load_state_dict(resume_iteration)
        iteration = checkpoint['iteration']
    else:
        iteration = 0
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)
    if 'cuda' in str(device):
        model.to(device)
    time1 = time.time()
    for batch_data_dict in train_loader:
        """batch_data_dict: {
        'audio_name': (batch_size [*2 if mixup],),
        'waveform': (batch_size [*2 if mixup], clip_samples),
        'target': (batch_size [*2 if mixup], classes_num),
        (ifexist) 'mixup_lambda': (batch_size * 2,)}
        """
        if (iteration % 2000 == 0 and iteration > resume_iteration or 
            iteration == 0):
            train_fin_time = time.time()
            bal_statistics = evaluator.evaluate(eval_bal_loader)
            test_statistics = evaluator.evaluate(eval_test_loader)
            logging.info('Validate bal mAP: {:.3f}'.format(np.mean(
                bal_statistics['average_precision'])))
            logging.info('Validate test mAP: {:.3f}'.format(np.mean(
                test_statistics['average_precision'])))
            statistics_container.append(iteration, bal_statistics,
                data_type='bal')
            statistics_container.append(iteration, test_statistics,
                data_type='test')
            statistics_container.dump()
            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time
            logging.info(
                'iteration: {}, train time: {:.3f} s, validate time: {:.3f} s'
                .format(iteration, train_time, validate_time))
            logging.info('------------------------------------')
            train_bgn_time = time.time()
        if iteration % 100000 == 0:
            checkpoint = {'iteration': iteration, 'model': model.module.
                state_dict(), 'sampler': train_sampler.state_dict()}
            checkpoint_path = os.path.join(checkpoints_dir,
                '{}_iterations.pth'.format(iteration))
            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))
        if 'mixup' in augmentation:
            batch_data_dict['mixup_lambda'] = mixup_augmenter.get_lambda(
                batch_size=len(batch_data_dict['waveform']))
        for key in batch_data_dict.keys():
            batch_data_dict[key] = move_data_to_device(batch_data_dict[key],
                device)
        model.train()
        if 'mixup' in augmentation:
            batch_output_dict = model(batch_data_dict['waveform'],
                batch_data_dict['mixup_lambda'])
            """{'clipwise_output': (batch_size, classes_num), ...}"""
            batch_target_dict = {'target': do_mixup(batch_data_dict[
                'target'], batch_data_dict['mixup_lambda'])}
            """{'target': (batch_size, classes_num)}"""
        else:
            batch_output_dict = model(batch_data_dict['waveform'], None)
            """{'clipwise_output': (batch_size, classes_num), ...}"""
            batch_target_dict = {'target': batch_data_dict['target']}
            """{'target': (batch_size, classes_num)}"""
        loss = loss_func(batch_output_dict, batch_target_dict)
        loss.backward()
        print(loss)
        optimizer.step()
        optimizer.zero_grad()
        if iteration % 10 == 0:
            print('--- Iteration: {}, train time: {:.3f} s / 10 iterations ---'
                .format(iteration, time.time() - time1))
            time1 = time.time()
        if iteration == early_stop:
            break
        iteration += 1
