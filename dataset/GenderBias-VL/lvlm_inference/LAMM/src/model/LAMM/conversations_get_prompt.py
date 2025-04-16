def get_prompt(self):
    if self.sep_style == SeparatorStyle.SINGLE:
        ret = self.system + self.sep
        for role, message in self.messages:
            if message:
                if type(message) is tuple:
                    message, _, _ = message
                ret += role + ': ' + message + self.sep
            else:
                ret += role + ':'
        return ret
    elif self.sep_style == SeparatorStyle.TWO:
        seps = [self.sep, self.sep2]
        ret = self.system + seps[0]
        for i, (role, message) in enumerate(self.messages):
            if message:
                if type(message) is tuple:
                    message, _, _ = message
                ret += role + ': ' + message + seps[i % 2]
            else:
                ret += role + ':'
        return ret
    if self.sep_style == SeparatorStyle.MPT:
        ret = self.system + self.sep
        for role, message in self.messages:
            if message:
                if type(message) is tuple:
                    message, _, _ = message
                ret += role + message + self.sep
            else:
                ret += role
        return ret
    else:
        raise ValueError(f'Invalid style: {self.sep_style}')
