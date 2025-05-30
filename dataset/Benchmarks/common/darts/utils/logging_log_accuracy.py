def log_accuracy(accuracy, split: str='train'):
    """Log the average accuracy

    Parameters
    ----------
    accuracy: darts.MultitaskAccuracyMeter
        Current accuracy meter state

    split: str
        Either training of testing
    """
    acc_info = (
        f">>> {split.upper()} Accuracy - Subsite: {accuracy.get_avg_accuracy('subsite'):.4f}, Laterality: {accuracy.get_avg_accuracy('laterality'):.4f}, Behavior: {accuracy.get_avg_accuracy('behavior'):.4f}, Grade: {accuracy.get_avg_accuracy('grade'):.4f}"
        )
    logger.info(acc_info)
