def close(self):
    if self.on_memory:
        pass
    else:
        self.store.close()
