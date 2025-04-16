def get_output(output, config, copy_vocab, tokenzier):
    _BOUNDARY = tokenzier.eos_token_id
    N, D = output.size()
    output = output.detach().cpu()
    out = []
    for i in range(N):
        txt = []
        for j in range(D):
            ix = output[i, j].item()
            if ix == _BOUNDARY:
                break
            if ix < config.vocab_size:
                txt.append(ix)
            else:
                ix = ix - config.vocab_size
                txt += copy_vocab.token_fg_w[ix]
        out.append(txt)
    return out
