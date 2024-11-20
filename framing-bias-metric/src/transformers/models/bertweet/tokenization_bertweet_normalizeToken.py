def normalizeToken(self, token):
    """
        Normalize tokens in a Tweet
        """
    lowercased_token = token.lower()
    if token.startswith('@'):
        return '@USER'
    elif lowercased_token.startswith('http') or lowercased_token.startswith(
        'www'):
        return 'HTTPURL'
    elif len(token) == 1:
        if token in self.special_puncts:
            return self.special_puncts[token]
        if self.demojizer is not None:
            return self.demojizer(token)
        else:
            return token
    else:
        return token
