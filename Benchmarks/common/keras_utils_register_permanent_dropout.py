def register_permanent_dropout():
    get_custom_objects()['PermanentDropout'] = PermanentDropout
