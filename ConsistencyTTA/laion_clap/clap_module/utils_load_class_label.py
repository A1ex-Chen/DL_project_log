def load_class_label(path):
    out = None
    if path is not None:
        if pathlib.Path(path).suffix in ['.pkl', '.pickle']:
            out = load_p(path)
        elif pathlib.Path(path).suffix in ['.json', '.txt']:
            out = load_json(path)
        elif pathlib.Path(path).suffix in ['.npy', '.npz']:
            out = np.load(path)
        elif pathlib.Path(path).suffix in ['.csv']:
            import pandas as pd
            out = pd.read_csv(path)
    return out
