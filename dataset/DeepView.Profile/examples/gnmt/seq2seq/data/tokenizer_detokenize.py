def detokenize(self, inputs, delim=' '):
    """
        Detokenizes single sentence and removes token separator characters.

        :param inputs: sequence of tokens
        :param delim: tokenization delimiter

        returns: string representing detokenized sentence
        """
    detok = delim.join([self.idx2tok[idx] for idx in inputs])
    detok = detok.replace(self.separator + ' ', '')
    detok = detok.replace(self.separator, '')
    detok = detok.replace(config.BOS_TOKEN, '')
    detok = detok.replace(config.EOS_TOKEN, '')
    detok = detok.replace(config.PAD_TOKEN, '')
    detok = detok.strip()
    return detok
