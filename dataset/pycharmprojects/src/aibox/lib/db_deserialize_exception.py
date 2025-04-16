@staticmethod
def deserialize_exception(serialized_exception: str) ->Exception:
    exception_dict = json.loads(serialized_exception)
    return DB.Log.Exception(code=exception_dict['code'], type=
        exception_dict['type'], message=exception_dict['message'],
        traceback=exception_dict['traceback'])
