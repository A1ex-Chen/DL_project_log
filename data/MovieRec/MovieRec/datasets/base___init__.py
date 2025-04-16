def __init__(self, args):
    self.args = args
    self.min_rating = args.min_rating
    self.min_uc = args.min_uc
    self.min_sc = args.min_sc
    self.split = args.split
    assert self.min_uc >= 2, 'Need at least 2 ratings per user for validation and test'
