def dump(self):
    pickle.dump(self.statistics_dict, open(self.statistics_path, 'wb'))
    pickle.dump(self.statistics_dict, open(self.backup_statistics_path, 'wb'))
    logging.info('    Dump statistics to {}'.format(self.statistics_path))
    logging.info('    Dump statistics to {}'.format(self.
        backup_statistics_path))
