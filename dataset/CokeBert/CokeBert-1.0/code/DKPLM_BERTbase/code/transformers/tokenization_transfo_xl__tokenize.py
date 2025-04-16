def _tokenize(self, line, add_eos=False, add_double_eos=False):
    line = line.strip()
    if self.lower_case:
        line = line.lower()
    if self.delimiter == '':
        symbols = line
    else:
        symbols = line.split(self.delimiter)
    if add_double_eos:
        return ['<S>'] + symbols + ['<S>']
    elif add_eos:
        return symbols + ['<eos>']
    else:
        return symbols
