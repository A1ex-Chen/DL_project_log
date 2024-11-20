def get_prompt(self) ->str:
    """Get the prompt for generation."""
    if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
        ret = self.system + self.sep
        for role, message in self.messages:
            if message:
                ret += role + ': ' + message + self.sep
            else:
                ret += role + ':'
        return ret
    elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
        seps = [self.sep, self.sep2]
        ret = self.system + seps[0]
        for i, (role, message) in enumerate(self.messages):
            if message:
                ret += role + ': ' + message + seps[i % 2]
            else:
                ret += role + ':'
        return ret
    elif self.sep_style == SeparatorStyle.ADD_SPACE_TWO:
        seps = [self.sep, self.sep2]
        ret = self.system + seps[0]
        for i, (role, message) in enumerate(self.messages):
            if message:
                ret += role + ' ' + message + seps[i % 2]
            else:
                ret += role + ''
        return ret
    elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
        ret = self.system
        for role, message in self.messages:
            if message:
                ret += role + message + self.sep
            else:
                ret += role
        return ret
    elif self.sep_style == SeparatorStyle.BAIZE:
        ret = self.system + '\n'
        for role, message in self.messages:
            if message:
                ret += role + message + '\n'
            else:
                ret += role
        return ret
    elif self.sep_style == SeparatorStyle.DOLLY:
        seps = [self.sep, self.sep2]
        ret = self.system
        for i, (role, message) in enumerate(self.messages):
            if message:
                ret += role + ':\n' + message + seps[i % 2]
                if i % 2 == 1:
                    ret += '\n\n'
            else:
                ret += role + ':\n'
        return ret
    elif self.sep_style == SeparatorStyle.RWKV:
        ret = self.system
        for i, (role, message) in enumerate(self.messages):
            if message:
                ret += role + ': ' + message.replace('\r\n', '\n').replace(
                    '\n\n', '\n')
                ret += '\n\n'
            else:
                ret += role + ':'
        return ret
    elif self.sep_style == SeparatorStyle.PHOENIX:
        ret = self.system
        for role, message in self.messages:
            if message:
                ret += role + ': ' + '<s>' + message + '</s>'
            else:
                ret += role + ': ' + '<s>'
        return ret
    elif self.sep_style == SeparatorStyle.NEW_LINE:
        ret = self.system + self.sep
        for role, message in self.messages:
            if message:
                ret += role + '\n' + message + self.sep
            else:
                ret += role + '\n'
        return ret
    elif self.sep_style == SeparatorStyle.BILLA:
        ret = self.system + self.sep
        for role, message in self.messages:
            if message:
                ret += role + ': ' + message + self.sep
            else:
                ret += role + ': '
        return ret
    else:
        raise ValueError(f'Invalid style: {self.sep_style}')
