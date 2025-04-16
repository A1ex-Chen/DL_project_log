def log_single_accuracy(accuracy, split: str='train'):
    """Log the average accuracy for a single task

    Parameters
    ----------
    accuracy: darts.MultitaskAccuracyMeter
        Current accuracy meter state

    split: str
        Either training of testing
    """
    acc_info = (
        f">>> {split.upper()} Accuracy - Response: {accuracy.get_avg_accuracy('response'):.4f}, "
        )
    logger.info(acc_info)
