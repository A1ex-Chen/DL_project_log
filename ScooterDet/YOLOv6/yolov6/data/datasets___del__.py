def __del__(self):
    if self.cache_ram:
        del self.imgs
