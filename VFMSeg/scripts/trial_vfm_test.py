def test(cfg, args, output_dir=''):
    logger = logging.getLogger('xmuda.test')
    model_2d = build_model_2d(cfg)[0]
    model_3d = build_model_3d(cfg)[0]
    model_2d = model_2d.cuda()
    model_3d = model_3d.cuda()
    checkpointer_2d = CheckpointerV2(model_2d, save_dir=output_dir, logger=
        logger)
    if args.ckpt2d:
        weight_path = args.ckpt2d.replace('@', output_dir)
        checkpointer_2d.load(weight_path, resume=False)
    else:
        checkpointer_2d.load(None, resume=True)
    checkpointer_3d = CheckpointerV2(model_3d, save_dir=output_dir, logger=
        logger)
    if args.ckpt3d:
        weight_path = args.ckpt3d.replace('@', output_dir)
        checkpointer_3d.load(weight_path, resume=False)
    else:
        checkpointer_3d.load(None, resume=True)
    test_dataloader = build_dataloader(cfg, args, mode='test', domain='target')
    pselab_path = None
    if args.pselab:
        pselab_dir = osp.join(output_dir, 'pselab_data')
        os.makedirs(pselab_dir, exist_ok=True)
        assert len(cfg.DATASET_TARGET.TEST) == 1
        pselab_path = osp.join(pselab_dir, cfg.DATASET_TARGET.TEST[0] + '.npy')
    set_random_seed(cfg.RNG_SEED)
    test_metric_logger = MetricLogger(delimiter='  ')
    model_2d.eval()
    model_3d.eval()
    if args.pselab:
        predict(cfg, args, model_2d, model_3d, test_dataloader, pselab_path,
            args.save_ensemble)
    else:
        validate(cfg, args, model_2d, model_3d, test_dataloader,
            test_metric_logger, pselab_path=pselab_path)
