def remove_cache(self, param):
    if self.has_cache and param in self.cache:
        del self.cache[param]
