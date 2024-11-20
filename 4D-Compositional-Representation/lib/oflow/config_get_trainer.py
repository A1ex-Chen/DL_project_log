def get_trainer(model, optimizer, cfg, device, **kwargs):
    """ Returns an OFlow trainer instance.

    Args:
        model (nn.Module): OFlow model
        optimizer (optimizer): PyTorch optimizer
        cfg (yaml config): yaml config object
        device (device): PyTorch device
    """
    threshold = cfg['test']['threshold']
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
    input_type = cfg['data']['input_type']
    loss_corr = cfg['model']['loss_corr']
    loss_recon = cfg['model']['loss_recon']
    loss_corr_bw = cfg['model']['loss_corr_bw']
    eval_sample = cfg['training']['eval_sample']
    vae_beta = cfg['model']['vae_beta']
    trainer = training.Trainer(model, optimizer, device=device, input_type=
        input_type, vis_dir=vis_dir, threshold=threshold, eval_sample=
        eval_sample, loss_corr=loss_corr, loss_recon=loss_recon,
        loss_corr_bw=loss_corr_bw, vae_beta=vae_beta)
    return trainer
