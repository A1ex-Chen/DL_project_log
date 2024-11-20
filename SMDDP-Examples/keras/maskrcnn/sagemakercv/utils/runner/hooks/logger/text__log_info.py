def _log_info(self, log_dict, runner):
    if runner.mode == 'train':
        log_str = 'Epoch [{}][{}/{}]\tlr: {:.5f}, '.format(log_dict['epoch'
            ], log_dict['iter'], runner.num_examples, log_dict['lr'])
        if 'time' in log_dict.keys():
            eta_sec = log_dict['time'] * (runner.max_iters - runner.iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            log_str += 'eta: {}, '.format(eta_str)
            log_str += 'step time: {:.3f}, '.format(log_dict['time'])
    else:
        log_str = 'Epoch({}) [{}][{}]\t'.format(log_dict['mode'], log_dict[
            'epoch'] - 1, log_dict['iter'])
    log_items = []
    for name, val in log_dict.items():
        if name in ['mode', 'Epoch', 'iter', 'lr', 'time', 'data_time',
            'memory', 'epoch']:
            continue
        if isinstance(val, float):
            val = '{:.4f}'.format(val)
        log_items.append('{}: {}'.format(name, val))
    log_str += ', '.join(log_items)
    runner.logger.info(log_str)
