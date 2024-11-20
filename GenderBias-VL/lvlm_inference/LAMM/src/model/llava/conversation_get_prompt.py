def get_prompt(self):
    messages = self.messages
    if len(messages) > 0 and type(messages[0][1]) is tuple:
        messages = self.messages.copy()
        init_role, init_msg = messages[0].copy()
        init_msg = init_msg[0].replace('<image>', '').strip()
        if 'mmtag' in self.version:
            messages[0] = init_role, init_msg
            messages.insert(0, (self.roles[0], '<Image><image></Image>'))
            messages.insert(1, (self.roles[1], 'Received.'))
        else:
            messages[0] = init_role, '<image>\n' + init_msg
    if self.sep_style == SeparatorStyle.SINGLE:
        ret = self.system + self.sep
        for role, message in messages:
            if message:
                if type(message) is tuple:
                    message, _, _ = message
                ret += role + ': ' + message + self.sep
            else:
                ret += role + ':'
    elif self.sep_style == SeparatorStyle.TWO:
        seps = [self.sep, self.sep2]
        ret = self.system + seps[0]
        for i, (role, message) in enumerate(messages):
            if message:
                if type(message) is tuple:
                    message, _, _ = message
                ret += role + ': ' + message + seps[i % 2]
            else:
                ret += role + ':'
    elif self.sep_style == SeparatorStyle.MPT:
        ret = self.system + self.sep
        for role, message in messages:
            if message:
                if type(message) is tuple:
                    message, _, _ = message
                ret += role + message + self.sep
            else:
                ret += role
    elif self.sep_style == SeparatorStyle.LLAMA_2:
        wrap_sys = lambda msg: f'<<SYS>>\n{msg}\n<</SYS>>\n\n'
        wrap_inst = lambda msg: f'[INST] {msg} [/INST]'
        ret = ''
        for i, (role, message) in enumerate(messages):
            if i == 0:
                assert message, 'first message should not be none'
                assert role == self.roles[0
                    ], 'first message should come from user'
            if message:
                if type(message) is tuple:
                    message, _, _ = message
                if i == 0:
                    message = wrap_sys(self.system) + message
                if i % 2 == 0:
                    message = wrap_inst(message)
                    ret += self.sep + message
                else:
                    ret += ' ' + message + ' ' + self.sep2
            else:
                ret += ''
        ret = ret.lstrip(self.sep)
    elif self.sep_style == SeparatorStyle.PLAIN:
        seps = [self.sep, self.sep2]
        ret = self.system
        for i, (role, message) in enumerate(messages):
            if message:
                if type(message) is tuple:
                    message, _, _ = message
                ret += message + seps[i % 2]
            else:
                ret += ''
    else:
        raise ValueError(f'Invalid style: {self.sep_style}')
    return ret
