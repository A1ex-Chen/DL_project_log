def assert_equal(self, name, doc, update_ref=False):
    if update_ref or os.environ.get('UPDATE_REF') == '1':
        self.save(name, doc)
    ref = self.load(name)
    self._diff(name, ref, doc)
