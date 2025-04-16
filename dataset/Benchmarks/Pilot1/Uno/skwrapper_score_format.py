def score_format(metric, score, signed=False, eol=''):
    if signed:
        return '{:<25} = {:+.5f}'.format(metric, score) + eol
    else:
        return '{:<25} =  {:.5f}'.format(metric, score) + eol
