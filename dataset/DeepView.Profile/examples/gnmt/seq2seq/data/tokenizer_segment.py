def segment(self, line):
    """
        Tokenizes single sentence and adds special BOS and EOS tokens.

        :param line: sentence

        returns: list representing tokenized sentence
        """
    line = line.strip().split()
    entry = [self.tok2idx[i] for i in line]
    entry = [config.BOS] + entry + [config.EOS]
    return entry
