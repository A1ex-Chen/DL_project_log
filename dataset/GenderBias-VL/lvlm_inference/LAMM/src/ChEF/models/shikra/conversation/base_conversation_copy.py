def copy(self):
    return Conversation(name=self.name, system=self.system, roles=self.
        roles, messages=[[x, y] for x, y in self.messages], offset=self.
        offset, sep_style=self.sep_style, sep=self.sep, sep2=self.sep2,
        stop_str=self.stop_str, stop_token_ids=self.stop_token_ids, conv_id
        =self.conv_id, model_name=self.model_name)
