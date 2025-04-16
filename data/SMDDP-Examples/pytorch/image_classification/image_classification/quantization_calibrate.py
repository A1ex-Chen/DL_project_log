def calibrate(model, train_loader, logger, calib_iter=1, percentile=99.99):
    """Calibrates whole network i.e. gathers data for quantization and calculates quantization parameters"""
    model.eval()
    with torch.no_grad():
        collect_stats(model, train_loader, logger, num_batches=calib_iter)
        compute_amax(model, method='percentile', percentile=percentile)
    logging.disable(logging.NOTSET)
