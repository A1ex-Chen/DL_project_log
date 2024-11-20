def create_hparam_dict(self):
    hparams = {'self': self.hparam_search_space} if hasattr(self,
        'hparam_search_space') else {}
    for attribute in vars(self):
        value = getattr(self, attribute)
        if hasattr(value, 'hparam_search_space'
            ) and attribute not in self.hparam_search_space:
            hparams[attribute] = {}
            search_space = value.hparam_search_space
            for param_name, param_space in search_space.items():
                hparams[attribute][param_name] = {'type': param_space[
                    'type'], 'values': param_space.get('values'), 'low':
                    param_space.get('low'), 'high': param_space.get('high')}
    return hparams
