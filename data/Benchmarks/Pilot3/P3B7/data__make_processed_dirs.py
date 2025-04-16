def _make_processed_dirs(self):
    processed = self.root.joinpath('processed')
    processed.joinpath('train').mkdir(parents=True)
    processed.joinpath('valid').mkdir()
