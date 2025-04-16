@property
def occupancies(self):
    return self.values < self.threshold
