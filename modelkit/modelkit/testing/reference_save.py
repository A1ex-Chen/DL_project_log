def save(self, name, doc):
    path = os.path.join(self.path, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fp:
        self._save(doc, fp)
