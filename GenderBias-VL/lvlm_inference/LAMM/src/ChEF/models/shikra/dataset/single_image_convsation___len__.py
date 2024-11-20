def __len__(self):
    self.initialize_if_needed()
    return len(self.dataset)
