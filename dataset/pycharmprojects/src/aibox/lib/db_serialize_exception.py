@staticmethod
def serialize_exception(exception: Exception) ->str:
    return json.dumps({'code': exception.code, 'type': exception.type,
        'message': exception.message, 'traceback': exception.traceback})
