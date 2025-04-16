def _save(self, save_path, epoch, metric=None, metric_ema=None):
    save_state = {'epoch': epoch, 'arch': type(self.model).__name__.lower(),
        'state_dict': get_state_dict(self.model, self.unwrap_fn), 'version': 2}
    if self.args is not None:
        save_state['arch'] = self.args.model
        save_state['args'] = self.args
    if self.model_ema is not None:
        if metric_ema > metric:
            save_state['state_dict'] = get_state_dict(self.model_ema, self.
                unwrap_fn)
    if metric is not None:
        save_state['metric'] = metric
    torch.save(save_state, save_path)
