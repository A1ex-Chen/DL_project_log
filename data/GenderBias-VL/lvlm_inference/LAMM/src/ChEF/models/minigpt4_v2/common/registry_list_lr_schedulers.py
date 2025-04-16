@classmethod
def list_lr_schedulers(cls):
    return sorted(cls.mapping['lr_scheduler_name_mapping'].keys())
