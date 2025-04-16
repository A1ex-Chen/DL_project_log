@classmethod
def get_lr_scheduler_class(cls, name):
    return cls.mapping['lr_scheduler_name_mapping'].get(name, None)
