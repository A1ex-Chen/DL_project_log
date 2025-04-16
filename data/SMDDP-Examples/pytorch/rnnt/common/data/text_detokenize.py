def detokenize(self, inds):
    if self.use_sentpiece:
        return self.sentpiece.decode(inds)
    else:
        return ''.join(self.charset[i] for i in inds)
