def save_vocabulary(self, save_directory: str, filename_prefix: Optional[
    str]=None) ->Tuple[str]:
    save_dir = Path(save_directory)
    assert save_dir.is_dir(), f'{save_directory} should be a directory'
    save_json(self.encoder, save_dir / ((filename_prefix + '-' if
        filename_prefix else '') + self.vocab_files_names['vocab']))
    for orig, f in zip(['source.spm', 'target.spm'], self.spm_files):
        dest_path = save_dir / ((filename_prefix + '-' if filename_prefix else
            '') + Path(f).name)
        if not dest_path.exists():
            copyfile(f, save_dir / orig)
    return tuple(save_dir / ((filename_prefix + '-' if filename_prefix else
        '') + f) for f in self.vocab_files_names)
