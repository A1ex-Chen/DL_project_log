def verify_results(cfg, results):
    """
    Args:
        results (OrderedDict[dict]): task_name -> {metric -> score}

    Returns:
        bool: whether the verification succeeds or not
    """
    expected_results = cfg.TEST.EXPECTED_RESULTS
    if not len(expected_results):
        return True
    ok = True
    for task, metric, expected, tolerance in expected_results:
        actual = results[task].get(metric, None)
        if actual is None:
            ok = False
            continue
        if not np.isfinite(actual):
            ok = False
            continue
        diff = abs(actual - expected)
        if diff > tolerance:
            ok = False
    logger = logging.getLogger(__name__)
    if not ok:
        logger.error('Result verification failed!')
        logger.error('Expected Results: ' + str(expected_results))
        logger.error('Actual Results: ' + pprint.pformat(results))
        sys.exit(1)
    else:
        logger.info('Results verification passed.')
    return ok
