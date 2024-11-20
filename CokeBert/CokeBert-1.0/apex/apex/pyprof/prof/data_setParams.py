def setParams(self, params):
    qaz = ''
    for key, value in params.items():
        if 'type' not in key:
            qaz += '{}={},'.format(key, value)
        elif type(value) is str:
            qaz += '{},'.format(Utility.typeToString(value))
        else:
            qaz += '{}'.format(value)
    self.params = qaz.replace(' ', '')
