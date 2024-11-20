def _train(config: Config, terminator: Event):
    path_to_checkpoints_dir = config.path_to_checkpoints_dir
    logger = Logger.build(name=os.path.basename(path_to_checkpoints_dir),
        path_to_log_file=os.path.join(path_to_checkpoints_dir, 'train.log'))
    logger.i(f'Created checkpoints directory: {path_to_checkpoints_dir}')
    logger.i(config.describe())
    logger.i('Packages:')
    for package in pkg_resources.working_set:
        logger.i(f'\t{package.project_name}, {package.version}')
    logger.i('Arguments:\n' + ' '.join(sys.argv[1:]))
    if config.task_name == Task.Name.CLASSIFICATION:
        from aibox.lib.task.classification.db import DB as T
        db_class = T
    elif config.task_name == Task.Name.DETECTION:
        from aibox.lib.task.detection.db import DB as T
        db_class = T
    elif config.task_name == Task.Name.INSTANCE_SEGMENTATION:
        from aibox.lib.task.instance_segmentation.trainer import Trainer
        assert isinstance(config, InstanceSegmentationConfig)
    else:
        raise ValueError
    db = db_class(path_to_db=os.path.join(path_to_checkpoints_dir,
        'summary.db'))
    db.insert_log_table(DB.Log(global_batch=0, status=DB.Log.Status.
        INITIALIZING, datetime=int(time.time()), epoch=0, total_epoch=
        'test', batch=0, total_batch=0, avg_loss=-1, learning_rate=-1,
        samples_per_sec=-1, eta_hrs=-1))
    try:
        torch.multiprocessing.set_sharing_strategy('file_system')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_count = 1 if not torch.cuda.is_available(
            ) else torch.cuda.device_count()
        augmenter = Augmenter(config.aug_strategy, config.aug_hflip_prob,
            config.aug_vflip_prob, config.aug_rotate90_prob, config.
            aug_crop_prob_and_min_max, config.aug_zoom_prob_and_min_max,
            config.aug_scale_prob_and_min_max, config.
            aug_translate_prob_and_min_max, config.
            aug_rotate_prob_and_min_max, config.aug_shear_prob_and_min_max,
            config.aug_blur_prob_and_min_max, config.
            aug_sharpen_prob_and_min_max, config.aug_color_prob_and_min_max,
            config.aug_brightness_prob_and_min_max, config.
            aug_grayscale_prob_and_min_max, config.
            aug_contrast_prob_and_min_max, config.
            aug_noise_prob_and_min_max, config.
            aug_resized_crop_prob_and_width_height)
        if config.task_name == Task.Name.CLASSIFICATION:
            from aibox.lib.task.classification.trainer import Trainer
            assert isinstance(config, ClassificationConfig)
        elif config.task_name == Task.Name.DETECTION:
            from aibox.lib.task.detection.trainer import Trainer
            assert isinstance(config, DetectionConfig)
        elif config.task_name == Task.Name.INSTANCE_SEGMENTATION:
            from aibox.lib.task.instance_segmentation.trainer import Trainer
            assert isinstance(config, InstanceSegmentationConfig)
        else:
            raise ValueError('Invalid task name')
        trainer = Trainer(config, logger, augmenter, device, device_count,
            db, terminator, path_to_checkpoints_dir)
        logger.i('Start training with {:s} (batch size: {:d})'.format('CPU' if
            device == torch.device('cpu') else '{:d} GPUs'.format(
            device_count), config.batch_size))
        time_checkpoint = time.time()
        trainer.train()
        elapsed_time = time.time() - time_checkpoint
        logger.i('Done! Elapsed {:.2f} hrs'.format(elapsed_time / 3600))
    except Exception as e:
        logger.ex(f'Something went wrong')
        exception_class = e.__class__
        if exception_class == RuntimeError:
            exception_code = 'E01'
        elif exception_class == AssertionError:
            exception_code = 'E02'
        elif exception_class == ValueError:
            exception_code = 'E03'
        elif exception_class == FileNotFoundError:
            exception_code = 'E04'
        else:
            exception_code = 'E00'
        exception = DB.Log.Exception(code=exception_code, type=
            exception_class.__name__, message=str(e), traceback=traceback.
            format_exc())
        db.update_log_table_latest_exception(exception)
        raise e
    finally:
        db.close()
