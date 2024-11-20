def copy(self):
    return Conversation(system=self.system, roles=self.roles, messages=[[x,
        y] for x, y in self.messages], offset=self.offset, sep_style=self.
        sep_style, sep=self.sep, sep2=self.sep2, version=self.version)
