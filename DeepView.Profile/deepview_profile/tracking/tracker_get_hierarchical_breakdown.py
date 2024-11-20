def get_hierarchical_breakdown(self):
    if (self._weight_tracker is None or self._activations_tracker is None or
        self._peak_usage_bytes is None or self._operation_tracker is None):
        raise RuntimeError(
            'Memory tracking and run time tracking have not both been performed yet.'
            )
    return HierarchicalBreakdownBuilder().for_model(self._model
        ).set_peak_usage_bytes(self._peak_usage_bytes).process_tracker(self
        ._operation_tracker).process_tracker(self._activations_tracker
        ).process_tracker(self._weight_tracker).build()
