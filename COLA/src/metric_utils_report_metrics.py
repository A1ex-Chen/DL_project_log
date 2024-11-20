def report_metrics(res):
    df, epss, accs_bar, accs_l2, accs_pdy, accs_palldy, accs_pxy = res
    print(f'# Instances: {df.shape[0]}\t# eps grid: {len(epss)}')
    print(f'\tBest L1: {np.max(accs_bar)} at {epss[np.argmax(accs_bar)]}')
    print(f'\tBest L2: {np.max(accs_l2)} at {epss[np.argmax(accs_l2)]}')
    print(f'\tTemporal: {np.max(accs_pdy)}')
    print(f'\tUnadj: {np.max(accs_palldy)}')
    print(f'\tMisspecified: {np.max(accs_pxy)}')
