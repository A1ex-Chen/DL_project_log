def run(parameters, opt):
    hyp_dict = {k: v for k, v in parameters.items() if k not in ['epochs',
        'batch_size']}
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name,
        exist_ok=opt.exist_ok or opt.evolve))
    opt.batch_size = parameters.get('batch_size')
    opt.epochs = parameters.get('epochs')
    device = select_device(opt.device, batch_size=opt.batch_size)
    train(hyp_dict, opt, device, callbacks=Callbacks())
