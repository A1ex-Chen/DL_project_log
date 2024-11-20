def build_conv(self, source: List[Dict[str, Any]]) ->Conversation:
    conv = self.conv_template()
    role_map = {'human': conv.roles[0], 'gpt': conv.roles[1]}
    assert len(source) > 0
    assert source[0]['from'] == 'human'
    for sentence in source:
        role = role_map[sentence['from']]
        conv.append_message(role, sentence['value'])
    return conv
