def __init__(self, statistics_path):
    """Contain statistics of different training iterations."""
    self.statistics_path = statistics_path
    self.backup_statistics_path = '{}_{}.pkl'.format(os.path.splitext(self.
        statistics_path)[0], datetime.datetime.now().strftime(
        '%Y-%m-%d_%H-%M-%S'))
    self.statistics_dict = {'bal': [], 'test': []}
