@classmethod
def flatten(cls, obj):
    return (obj.tensor,), cls(_convert_target_to_string(type(obj)))
