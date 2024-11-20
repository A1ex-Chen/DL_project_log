def rewrite_logs(d):
    new_d = {}
    eval_prefix = 'eval_'
    eval_prefix_len = len(eval_prefix)
    for k, v in d.items():
        if k.startswith(eval_prefix):
            new_d['eval/' + k[eval_prefix_len:]] = v
        else:
            new_d['train/' + k] = v
    return new_d
