def _load_file(self, path):
    return torch.load(path, map_location=torch.device('cpu'))
