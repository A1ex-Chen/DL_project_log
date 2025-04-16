@property
def results_dict(self):
    """Returns dictionary of computed performance metrics and statistics."""
    return dict(zip(self.keys + ['fitness'], self.mean_results() + [self.
        fitness]))
