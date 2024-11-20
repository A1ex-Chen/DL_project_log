def symlink(self, src, dst):
    """Like os.symlink, but overwrites dst and logs"""
    self.debug("linking: '%s' -> '%s'" % (self.relpath(dst), self.relpath(src))
        )
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(src, dst)
