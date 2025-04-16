def promptsmaker(prompts, batch):
    out_p = []
    plen = len(prompts)
    for prompt in prompts:
        add = ''
        if KCOMM in prompt:
            add, prompt = prompt.split(KCOMM)
            add = add + ' '
        prompts = prompt.split(KBRK)
        out_p.append([(add + p) for p in prompts])
    out = [None] * batch * len(out_p[0]) * len(out_p)
    for p, prs in enumerate(out_p):
        for r, pr in enumerate(prs):
            start = (p + r * plen) * batch
            out[start:start + batch] = [pr] * batch
    return out, out_p
