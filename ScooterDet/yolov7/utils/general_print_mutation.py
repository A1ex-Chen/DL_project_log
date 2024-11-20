def print_mutation(hyp, results, yaml_file='hyp_evolved.yaml', bucket=''):
    a = '%10s' * len(hyp) % tuple(hyp.keys())
    b = '%10.3g' * len(hyp) % tuple(hyp.values())
    c = '%10.4g' * len(results) % results
    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))
    if bucket:
        url = 'gs://%s/evolve.txt' % bucket
        if gsutil_getsize(url) > (os.path.getsize('evolve.txt') if os.path.
            exists('evolve.txt') else 0):
            os.system('gsutil cp %s .' % url)
    with open('evolve.txt', 'a') as f:
        f.write(c + b + '\n')
    x = np.unique(np.loadtxt('evolve.txt', ndmin=2), axis=0)
    x = x[np.argsort(-fitness(x))]
    np.savetxt('evolve.txt', x, '%10.3g')
    for i, k in enumerate(hyp.keys()):
        hyp[k] = float(x[0, i + 7])
    with open(yaml_file, 'w') as f:
        results = tuple(x[0, :7])
        c = '%10.4g' * len(results) % results
        f.write(
            '# Hyperparameter Evolution Results\n# Generations: %g\n# Metrics: '
             % len(x) + c + '\n\n')
        yaml.dump(hyp, f, sort_keys=False)
    if bucket:
        os.system('gsutil cp evolve.txt %s gs://%s' % (yaml_file, bucket))
