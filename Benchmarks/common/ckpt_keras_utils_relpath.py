def relpath(self, p):
    return p.relative_to(self.cwd)
