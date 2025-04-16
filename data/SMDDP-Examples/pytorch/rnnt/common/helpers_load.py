def load(self, fpath, model, ema_model, optimizer, meta):
    print_once(f'Loading model from {fpath}')
    checkpoint = torch.load(fpath, map_location='cpu')
    unwrap_ddp = lambda model: getattr(model, 'module', model)
    state_dict = checkpoint['state_dict']
    unwrap_ddp(model).load_state_dict(state_dict, strict=False)
    if ema_model is not None:
        if checkpoint.get('ema_state_dict') is not None:
            key = 'ema_state_dict'
        else:
            key = 'state_dict'
            print_once('WARNING: EMA weights not found in the checkpoint.')
            print_once('WARNING: Initializing EMA model with regular params.')
        state_dict = checkpoint[key]
        unwrap_ddp(ema_model).load_state_dict(state_dict, strict=False)
    optimizer.load_state_dict(checkpoint['optimizer'])
    if self.use_amp:
        amp.load_state_dict(checkpoint['amp'])
    meta['start_epoch'] = checkpoint.get('epoch')
    meta['best_wer'] = checkpoint.get('best_wer', meta['best_wer'])
