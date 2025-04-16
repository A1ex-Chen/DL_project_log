@staticmethod
def deserialize_metrics(serialized_metrics: str) ->Metrics:
    metric_dict = json.loads(serialized_metrics)
    return DB.Checkpoint.Metrics(**metric_dict)
