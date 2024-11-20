def drop(self):
    if self.survival_prob == 1.0:
        return False
    return random.uniform(0.0, 1.0) > self.survival_prob
