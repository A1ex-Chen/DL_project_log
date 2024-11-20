def debug_tensor(tensor, name):
    """
    Simple utility which helps with debugging.
    Takes a tensor and outputs: min, max, avg, std, number of NaNs, number of
    INFs.

    :param tensor: torch tensor
    :param name: name of the tensor (only for logging)
    """
    logging.info(name)
    tensor = tensor.detach().float().cpu().numpy()
    logging.info(
        f'MIN: {tensor.min()} MAX: {tensor.max()} AVG: {tensor.mean()} STD: {tensor.std()} NAN: {np.isnan(tensor).sum()} INF: {np.isinf(tensor).sum()}'
        )
