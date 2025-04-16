def main(opt, callbacks=Callbacks()):
    if RANK in {-1, 0}:
        print_args(vars(opt))
        check_git_status()
        check_requirements(ROOT / 'requirements.txt')
    if opt.resume and not check_comet_resume(opt) and not opt.evolve:
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else
            get_latest_run())
        opt_yaml = last.parent.parent / 'opt.yaml'
        opt_data = opt.data
        if opt_yaml.is_file():
            with open(opt_yaml, errors='ignore') as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location='cpu')['opt']
        opt = argparse.Namespace(**d)
        opt.cfg, opt.weights, opt.resume = '', str(last), True
        if is_url(opt_data):
            opt.data = check_file(opt_data)
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = check_file(opt
            .data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights
            ), str(opt.project)
        assert len(opt.cfg) or len(opt.weights
            ), 'either --cfg or --weights must be specified'
        if opt.evolve:
            if opt.project == str(ROOT / 'runs/train'):
                opt.project = str(ROOT / 'runs/evolve')
            opt.exist_ok, opt.resume = opt.resume, False
        if opt.name == 'cfg':
            opt.name = Path(opt.cfg).stem
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name,
            exist_ok=opt.exist_ok))
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
        assert not opt.image_weights, f'--image-weights {msg}'
        assert not opt.evolve, f'--evolve {msg}'
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count(
            ) > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend='nccl' if dist.is_nccl_available() else
            'gloo', timeout=timedelta(seconds=10800))
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)
    else:
        meta = {'lr0': (1, 1e-05, 0.1), 'lrf': (1, 0.01, 1.0), 'momentum':
            (0.3, 0.6, 0.98), 'weight_decay': (1, 0.0, 0.001),
            'warmup_epochs': (1, 0.0, 5.0), 'warmup_momentum': (1, 0.0, 
            0.95), 'warmup_bias_lr': (1, 0.0, 0.2), 'box': (1, 0.02, 0.2),
            'cls': (1, 0.2, 4.0), 'cls_pw': (1, 0.5, 2.0), 'obj': (1, 0.2, 
            4.0), 'obj_pw': (1, 0.5, 2.0), 'iou_t': (0, 0.1, 0.7),
            'anchor_t': (1, 2.0, 8.0), 'anchors': (2, 2.0, 10.0),
            'fl_gamma': (0, 0.0, 2.0), 'hsv_h': (1, 0.0, 0.1), 'hsv_s': (1,
            0.0, 0.9), 'hsv_v': (1, 0.0, 0.9), 'degrees': (1, 0.0, 45.0),
            'translate': (1, 0.0, 0.9), 'scale': (1, 0.0, 0.9), 'shear': (1,
            0.0, 10.0), 'perspective': (0, 0.0, 0.001), 'flipud': (1, 0.0, 
            1.0), 'fliplr': (0, 0.0, 1.0), 'mosaic': (1, 0.0, 1.0), 'mixup':
            (1, 0.0, 1.0), 'copy_paste': (1, 0.0, 1.0)}
        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)
            if 'anchors' not in hyp:
                hyp['anchors'] = 3
        if opt.noautoanchor:
            del hyp['anchors'], meta['anchors']
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)
        evolve_yaml, evolve_csv = (save_dir / 'hyp_evolve.yaml', save_dir /
            'evolve.csv')
        if opt.bucket:
            subprocess.run(['gsutil', 'cp', f'gs://{opt.bucket}/evolve.csv',
                str(evolve_csv)])
        for _ in range(opt.evolve):
            if evolve_csv.exists():
                parent = 'single'
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                n = min(5, len(x))
                x = x[np.argsort(-fitness(x))][:n]
                w = fitness(x) - fitness(x).min() + 1e-06
                if parent == 'single' or len(x) == 1:
                    x = x[random.choices(range(n), weights=w)[0]]
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()
                mp, s = 0.8, 0.2
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([meta[k][0] for k in hyp.keys()])
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.
                        random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):
                    hyp[k] = float(x[i + 7] * v[i])
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])
                hyp[k] = min(hyp[k], v[2])
                hyp[k] = round(hyp[k], 5)
            results = train(hyp.copy(), opt, device, callbacks)
            callbacks = Callbacks()
            keys = ('metrics/precision', 'metrics/recall',
                'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', 'val/box_loss',
                'val/obj_loss', 'val/cls_loss')
            print_mutation(keys, results, hyp.copy(), save_dir, opt.bucket)
        plot_evolve(evolve_csv)
        LOGGER.info(
            f"""Hyperparameter evolution finished {opt.evolve} generations
Results saved to {colorstr('bold', save_dir)}
Usage example: $ python train.py --hyp {evolve_yaml}"""
            )
