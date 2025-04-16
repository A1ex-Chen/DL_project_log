@staticmethod
def shortname_for_key(info, param_name):
    words = param_name.split('_')
    shortname_parts = [TrialShortNamer.shortname_for_word(info, word) for
        word in words]
    separators = ['', '_']
    for separator in separators:
        shortname = separator.join(shortname_parts)
        if shortname not in info['reverse_short_param']:
            info['short_param'][param_name] = shortname
            info['reverse_short_param'][shortname] = param_name
            return shortname
    return param_name
