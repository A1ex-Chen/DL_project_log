def __init__(self, train, val, test, user_count, item_count, sample_size,
    seed, save_folder):
    self.train = train
    self.val = val
    self.test = test
    self.user_count = user_count
    self.item_count = item_count
    self.sample_size = sample_size
    self.seed = seed
    self.save_folder = save_folder
