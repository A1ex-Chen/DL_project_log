def _load_data(self, split):
    split_path = self.root.joinpath(f'processed/{split}')
    self.data = torch.load(split_path.joinpath('data.pt'))
    self.target = torch.load(split_path.joinpath('target.pt'))
