def _initialize_extra_fields(self, instances: Instances) ->Instances:
    """
        If input instances don't have ID, ID_period, lost_frame_count fields,
        this method is used to initialize these fields.

        Args:
            instances: D2 Instances, for predictions of the current frame
        Return:
            D2 Instances with extra fields added
        """
    if not instances.has('ID'):
        instances.set('ID', [None] * len(instances))
    if not instances.has('ID_period'):
        instances.set('ID_period', [None] * len(instances))
    if not instances.has('lost_frame_count'):
        instances.set('lost_frame_count', [None] * len(instances))
    if self._prev_instances is None:
        instances.ID = list(range(len(instances)))
        self._id_count += len(instances)
        instances.ID_period = [1] * len(instances)
        instances.lost_frame_count = [0] * len(instances)
    return instances
