def get_data(dump_dir, prefix):
    """Loads and concatenates dump files for given prefix (ex. inputs, outputs, labels, ids)"""
    dump_dir = Path(dump_dir)
    npz_files = sorted(dump_dir.glob(f'{prefix}*.npz'))
    data = None
    if npz_files:
        names = list(np.load(npz_files[0].as_posix()).keys())
        target_shape = {name: tuple(np.max([np.load(npz_file.as_posix())[
            name].shape for npz_file in npz_files], axis=0)) for name in names}
        data = {name: np.concatenate([pad_except_batch_axis(np.load(
            npz_file.as_posix())[name], target_shape[name]) for npz_file in
            npz_files]) for name in names}
    return data
