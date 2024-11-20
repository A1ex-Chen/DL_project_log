def write_abs_stats(stats_file, alphas, accs, abst):
    abs_file = open(stats_file, 'a')
    for k in range(alphas.shape[0]):
        abs_file.write('%10.5e,' % K.get_value(alphas[k]))
    for k in range(alphas.shape[0]):
        abs_file.write('%10.5e,' % accs[k])
    for k in range(alphas.shape[0]):
        abs_file.write('%10.5e,' % abst[k])
    abs_file.write('\n')
