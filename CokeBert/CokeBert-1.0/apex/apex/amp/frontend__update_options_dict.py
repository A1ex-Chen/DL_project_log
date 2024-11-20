def _update_options_dict(new_options):
    for k, v in new_options:
        if k in self.options:
            self.options[k] = v
        else:
            raise ValueError('Tried to set unexpected option {}'.format(k))
