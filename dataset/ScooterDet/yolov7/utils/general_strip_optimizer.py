def strip_optimizer(f='best.pt', s=''):
    x = torch.load(f, map_location=torch.device('cpu'))
    if x.get('ema'):
        x['model'] = x['ema']
    for k in ('optimizer', 'training_results', 'wandb_id', 'ema', 'updates'):
        x[k] = None
    x['epoch'] = -1
    x['model'].half()
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1000000.0
    print(
        f"Optimizer stripped from {f},{' saved as %s,' % s if s else ''} {mb:.1f}MB"
        )
