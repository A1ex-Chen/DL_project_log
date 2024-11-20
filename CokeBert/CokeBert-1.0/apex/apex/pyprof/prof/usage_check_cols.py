def check_cols(value):
    valid = ['idx', 'seq', 'altseq', 'tid', 'layer', 'trace', 'dir', 'sub',
        'mod', 'op', 'kernel', 'params', 'sil', 'tc', 'device', 'stream',
        'grid', 'block', 'flops', 'bytes']
    cols = value.split(',')
    for col in cols:
        if col not in valid:
            raise argparse.ArgumentTypeError(
                '{} is not a valid column name. Valid column names are {}.'
                .format(col, ','.join(valid)))
    return cols
