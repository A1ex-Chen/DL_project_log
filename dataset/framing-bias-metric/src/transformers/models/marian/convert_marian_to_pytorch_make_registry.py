def make_registry(repo_path='Opus-MT-train/models'):
    if not (Path(repo_path) / 'fr-en' / 'README.md').exists():
        raise ValueError(
            f'repo_path:{repo_path} does not exist: You must run: git clone git@github.com:Helsinki-NLP/Opus-MT-train.git before calling.'
            )
    results = {}
    for p in Path(repo_path).iterdir():
        n_dash = p.name.count('-')
        if n_dash == 0:
            continue
        else:
            lns = list(open(p / 'README.md').readlines())
            results[p.name] = _parse_readme(lns)
    return [(k, v['pre-processing'], v['download'], v['download'][:-4] +
        '.test.txt') for k, v in results.items()]
