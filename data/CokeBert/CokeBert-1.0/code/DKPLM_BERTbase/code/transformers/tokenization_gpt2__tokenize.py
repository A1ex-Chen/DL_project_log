def _tokenize(self, text, add_prefix_space=False):
    """ Tokenize a string.
            Args:
                - add_prefix_space (boolean, default False):
                    Begin the sentence with at least one space to get invariance to word order in GPT-2 (and RoBERTa) tokenizers.
        """
    if add_prefix_space:
        text = ' ' + text
    bpe_tokens = []
    for token in re.findall(self.pat, text):
        if sys.version_info[0] == 2:
            token = ''.join(self.byte_encoder[ord(b)] for b in token)
        else:
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8')
                )
        bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(' ')
            )
    return bpe_tokens
