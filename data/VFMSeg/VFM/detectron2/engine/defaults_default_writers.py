def default_writers(output_dir: str, max_iter: Optional[int]=None):
    """
    Build a list of :class:`EventWriter` to be used.
    It now consists of a :class:`CommonMetricPrinter`,
    :class:`TensorboardXWriter` and :class:`JSONWriter`.

    Args:
        output_dir: directory to store JSON metrics and tensorboard events
        max_iter: the total number of iterations

    Returns:
        list[EventWriter]: a list of :class:`EventWriter` objects.
    """
    PathManager.mkdirs(output_dir)
    return [CommonMetricPrinter(max_iter), JSONWriter(os.path.join(
        output_dir, 'metrics.json')), TensorboardXWriter(output_dir)]
