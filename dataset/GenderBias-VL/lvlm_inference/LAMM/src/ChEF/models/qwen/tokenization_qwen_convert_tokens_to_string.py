def convert_tokens_to_string(self, tokens: List[Union[bytes, str]]) ->str:
    """
        Converts a sequence of tokens in a single string.
        """
    text = ''
    temp = b''
    for t in tokens:
        if isinstance(t, str):
            if temp:
                text += temp.decode('utf-8', errors=self.errors)
                temp = b''
            text += t
        elif isinstance(t, bytes):
            temp += t
        else:
            raise TypeError('token should only be of type types or str')
    if temp:
        text += temp.decode('utf-8', errors=self.errors)
    return text
