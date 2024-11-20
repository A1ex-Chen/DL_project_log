def get_indexes(self):
    """Get a random index from the dataset."""
    return random.randint(0, len(self.dataset) - 1)
