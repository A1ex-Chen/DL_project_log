def _sorted_checkpoints(self, checkpoint_prefix=PREFIX_CHECKPOINT_DIR,
    use_mtime=False) ->List[str]:
    ordering_and_checkpoint_path = []
    glob_checkpoints = [str(x) for x in Path(self.args.output_dir).glob(
        f'{checkpoint_prefix}-*')]
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(f'.*{checkpoint_prefix}-([0-9]+)', path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups
                    ()[0]), path))
    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    if self.state.best_model_checkpoint is not None:
        best_model_index = checkpoints_sorted.index(str(Path(self.state.
            best_model_checkpoint)))
        checkpoints_sorted[best_model_index], checkpoints_sorted[-1
            ] = checkpoints_sorted[-1], checkpoints_sorted[best_model_index]
    return checkpoints_sorted
