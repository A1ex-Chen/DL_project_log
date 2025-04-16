@staticmethod
def serialize_metrics(metrics: Metrics) ->str:
    return json.dumps({'overall': metrics.overall.__dict__, 'specific':
        metrics.specific.__dict__})
