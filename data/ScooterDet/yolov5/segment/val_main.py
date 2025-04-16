def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard',
        'thop'))
    if opt.task in ('train', 'val', 'test'):
        if opt.conf_thres > 0.001:
            LOGGER.warning(
                f'WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results'
                )
        if opt.save_hybrid:
            LOGGER.warning(
                'WARNING ⚠️ --save-hybrid returns high mAP from hybrid labels, not from predictions alone'
                )
        run(**vars(opt))
    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.
            weights]
        opt.half = torch.cuda.is_available() and opt.device != 'cpu'
        if opt.task == 'speed':
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)
        elif opt.task == 'study':
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'
                x, y = list(range(256, 1536 + 128, 128)), []
                for opt.imgsz in x:
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)
                np.savetxt(f, y, fmt='%10.4g')
            subprocess.run(['zip', '-r', 'study.zip', 'study_*.txt'])
            plot_val_study(x=x)
        else:
            raise NotImplementedError(
                f'--task {opt.task} not in ("train", "val", "test", "speed", "study")'
                )
