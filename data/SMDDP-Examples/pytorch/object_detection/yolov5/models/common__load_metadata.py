@staticmethod
def _load_metadata(f='path/to/meta.yaml'):
    with open(f, errors='ignore') as f:
        d = yaml.safe_load(f)
    return d['stride'], d['names']
