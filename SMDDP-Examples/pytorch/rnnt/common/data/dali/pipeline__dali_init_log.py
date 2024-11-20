@staticmethod
def _dali_init_log(args: dict):
    if not dist.is_initialized() or dist.is_initialized() and dist.get_rank(
        ) == 0:
        max_len = max([len(ii) for ii in args.keys()])
        fmt_string = '\t%' + str(max_len) + 's : %s'
        print('Initializing DALI with parameters:')
        for keyPair in sorted(args.items()):
            print(fmt_string % keyPair)
