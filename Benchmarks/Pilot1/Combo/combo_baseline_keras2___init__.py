def __init__(self, save_all_models=False):
    Callback.__init__(self)
    self.save_all_models = save_all_models
    get_custom_objects()['PermanentDropout'] = PermanentDropout
