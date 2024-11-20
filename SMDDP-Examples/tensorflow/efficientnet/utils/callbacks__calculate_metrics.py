def _calculate_metrics(self) ->MutableMapping[str, Any]:
    logs = {}
    if self._track_lr:
        logs['learning_rate'] = self._calculate_lr()
    return logs
