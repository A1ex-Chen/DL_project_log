def stdout_metric_format(metric, metadata, value):
    name = metadata.get('name', metric + ' : ')
    unit = metadata.get('unit', None)
    format = f"{{{metadata.get('format', '')}}}"
    fields = [name, format.format(value) if value is not None else value, unit]
    fields = [f for f in fields if f is not None]
    return '| ' + ' '.join(fields)
