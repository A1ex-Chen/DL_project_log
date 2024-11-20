def append(self, iteration, statistics, data_type):
    statistics['iteration'] = iteration
    self.statistics_dict[data_type].append(statistics)
