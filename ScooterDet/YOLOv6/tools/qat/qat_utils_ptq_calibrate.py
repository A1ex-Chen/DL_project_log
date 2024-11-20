def ptq_calibrate(model, train_loader, cfg):
    model.eval()
    model.cuda()
    with torch.no_grad():
        collect_stats(model, train_loader, cfg.ptq.calib_batches)
        compute_amax(model, method=cfg.ptq.histogram_amax_method,
            percentile=cfg.ptq.histogram_amax_percentile)
