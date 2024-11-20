def json_serializer(obj):
    if isinstance(obj, datetime.date):
        return obj.isoformat()
    elif isinstance(obj, decimal.Decimal):
        return str(obj)
    raise TypeError('Unexpected ' + obj.__class__.__name__)
