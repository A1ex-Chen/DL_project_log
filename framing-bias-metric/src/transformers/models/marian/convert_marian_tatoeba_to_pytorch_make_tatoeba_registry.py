@staticmethod
def make_tatoeba_registry(repo_path=DEFAULT_MODEL_DIR):
    if not (Path(repo_path) / 'zho-eng' / 'README.md').exists():
        raise ValueError(
            f'repo_path:{repo_path} does not exist: You must run: git clone git@github.com:Helsinki-NLP/Tatoeba-Challenge.git before calling.'
            )
    results = {}
    for p in Path(repo_path).iterdir():
        if len(p.name) != 7:
            continue
        lns = list(open(p / 'README.md').readlines())
        results[p.name] = _parse_readme(lns)
    return [(k, v['pre-processing'], v['download'], v['download'][:-4] +
        '.test.txt') for k, v in results.items()]
