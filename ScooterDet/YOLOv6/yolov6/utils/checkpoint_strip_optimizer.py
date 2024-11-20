def strip_optimizer(ckpt_dir, epoch):
    """Delete optimizer from saved checkpoint file"""
    for s in ['best', 'last']:
        ckpt_path = osp.join(ckpt_dir, '{}_ckpt.pt'.format(s))
        if not osp.exists(ckpt_path):
            continue
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        if ckpt.get('ema'):
            ckpt['model'] = ckpt['ema']
        for k in ['optimizer', 'ema', 'updates']:
            ckpt[k] = None
        ckpt['epoch'] = epoch
        ckpt['model'].half()
        for p in ckpt['model'].parameters():
            p.requires_grad = False
        torch.save(ckpt, ckpt_path)
