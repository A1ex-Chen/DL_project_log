def print_mutation(keys, results, hyp, save_dir, bucket, prefix=colorstr(
    'evolve: ')):
    evolve_csv = save_dir / 'evolve.csv'
    evolve_yaml = save_dir / 'hyp_evolve.yaml'
    keys = tuple(keys) + tuple(hyp.keys())
    keys = tuple(x.strip() for x in keys)
    vals = results + tuple(hyp.values())
    n = len(keys)
    if bucket:
        url = f'gs://{bucket}/evolve.csv'
        if gsutil_getsize(url) > (evolve_csv.stat().st_size if evolve_csv.
            exists() else 0):
            subprocess.run(['gsutil', 'cp', f'{url}', f'{save_dir}'])
    s = '' if evolve_csv.exists() else ('%20s,' * n % keys).rstrip(',') + '\n'
    with open(evolve_csv, 'a') as f:
        f.write(s + ('%20.5g,' * n % vals).rstrip(',') + '\n')
    with open(evolve_yaml, 'w') as f:
        data = pd.read_csv(evolve_csv, skipinitialspace=True)
        data = data.rename(columns=lambda x: x.strip())
        i = np.argmax(fitness(data.values[:, :4]))
        generations = len(data)
        f.write('# YOLOv5 Hyperparameter Evolution Results\n' +
            f'# Best generation: {i}\n' +
            f"""# Last generation: {generations - 1}
""" + '# ' + ', '.join
            (f'{x.strip():>20s}' for x in keys[:7]) + '\n' + '# ' + ', '.
            join(f'{x:>20.5g}' for x in data.values[i, :7]) + '\n\n')
        yaml.safe_dump(data.loc[i][7:].to_dict(), f, sort_keys=False)
    LOGGER.info(prefix +
        f'{generations} generations finished, current result:\n' + prefix +
        ', '.join(f'{x.strip():>20s}' for x in keys) + '\n' + prefix + ', '
        .join(f'{x:20.5g}' for x in vals) + '\n\n')
    if bucket:
        subprocess.run(['gsutil', 'cp', f'{evolve_csv}', f'{evolve_yaml}',
            f'gs://{bucket}'])
