def build_output_dict(iou, stats, verbose=False):
    if verbose:
        format_string = ['AP 0.50:0.95 all', 'AP 0.50 all', 'AP 0.75 all',
            'AP 0.50:0.95 small', 'AP 0.50:0.95 medium',
            'AP 0.50:0.95 large', 'AR 0.50:0.95 all', 'AR 0.50 all',
            'AR 0.75 all', 'AR 0.50:0.95 small', 'AR 0.50:0.95 medium',
            'AR 0.50:0.95 large']
        stat_dict = {'{0} {1}'.format(iou, i): j for i, j in zip(
            format_string, stats)}
    else:
        stat_dict = {'{0} AP 0.50:0.95 all'.format(iou): stats[0]}
    return stat_dict
