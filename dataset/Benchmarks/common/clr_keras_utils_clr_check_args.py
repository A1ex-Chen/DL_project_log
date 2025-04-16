def clr_check_args(args):
    req_keys = ['clr_mode', 'clr_base_lr', 'clr_max_lr', 'clr_gamma']
    keys_present = True
    for key in req_keys:
        if key not in args.keys():
            keys_present = False
    return keys_present
