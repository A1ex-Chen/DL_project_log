def get_fields(self):
    """like `get_fields` in the Instances object,
        but return each field in tensor representations"""
    ret = {}
    for k, v in self.batch_extra_fields.items():
        ret[k] = v
    return ret
