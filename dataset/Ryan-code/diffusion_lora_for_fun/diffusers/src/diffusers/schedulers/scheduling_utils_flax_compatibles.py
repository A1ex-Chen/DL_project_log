@property
def compatibles(self):
    """
        Returns all schedulers that are compatible with this scheduler

        Returns:
            `List[SchedulerMixin]`: List of compatible schedulers
        """
    return self._get_compatibles()
