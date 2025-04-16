@staticmethod
def add_new_param_name(info, param_name):
    short_name = TrialShortNamer.shortname_for_key(info, param_name)
    info['short_param'][param_name] = short_name
    info['reverse_short_param'][short_name] = param_name
