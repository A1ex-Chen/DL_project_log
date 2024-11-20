def _save_split(self, split, data, target):
    target = self._create_target(target)
    split_path = self.root.joinpath(f'processed/{split}')
    torch.save(data, split_path.joinpath('data.pt'))
    torch.save(target, split_path.joinpath('target.pt'))
