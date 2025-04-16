def _create_metrics(self, tasks, avg):
    """Create F1 metrics for each of the tasks

        Args:
            tasks: dictionary of tasks and their respective number
                of classes
            avg: either 'micro' or 'macro'
        """
    return {t: F1(c, average=avg) for t, c in tasks.items()}
