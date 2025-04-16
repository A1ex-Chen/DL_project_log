def _load(self):
    with open(self.asset_path) as f:
        self.data = json.load(f)
