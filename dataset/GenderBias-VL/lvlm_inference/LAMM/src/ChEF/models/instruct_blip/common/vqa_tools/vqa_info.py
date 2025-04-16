def info(self):
    """
        Print information about the VQA annotation file.
        :return:
        """
    for key, value in self.datset['info'].items():
        print('%s: %s' % (key, value))
