def tokenize(self, sent):
    """This tries to mimic multi-bleu-detok from Moses, and by extension mteval-v13b.
        Code taken directly from there and attempted rewrite into Python."""
    sent = re.sub('<skipped>', '', sent)
    sent = re.sub('-\\n', '', sent)
    sent = re.sub('\\n', ' ', sent)
    sent = re.sub('&quot;', '"', sent)
    sent = re.sub('&amp;', '&', sent)
    sent = re.sub('&lt;', '<', sent)
    sent = re.sub('&gt;', '>', sent)
    sent = ' ' + sent + ' '
    sent = re.sub('([\\{-\\~\\[-\\` -\\&\\(-\\+\\:-\\@\\/])', ' \\1 ', sent)
    sent = re.sub('([^0-9])([\\.,])', '\\1 \\2 ', sent)
    sent = re.sub('([\\.,])([^0-9])', ' \\1 \\2', sent)
    sent = re.sub('([0-9])(-)', '\\1 \\2 ', sent)
    sent = re.sub('\\s+', ' ', sent)
    sent = sent.strip()
    return sent.split(' ')
