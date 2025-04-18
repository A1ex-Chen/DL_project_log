def main(config):
    logger = config.get_logger('test')
    data_loader = getattr(module_data, config['data_loader']['type'])(config
        ['data_loader']['args']['data_dir'], batch_size=512, shuffle=False,
        validation_split=0.0, training=False, num_workers=2)
    model = config.init_obj('arch', module_arch)
    logger.info(model)
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size
    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({met.__name__: (total_metrics[i].item() / n_samples) for i,
        met in enumerate(metric_fns)})
    logger.info(log)
