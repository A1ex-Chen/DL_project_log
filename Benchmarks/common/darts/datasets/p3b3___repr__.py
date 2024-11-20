def __repr__(self):
    fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
    fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
    tmp = self.partition
    fmt_str += '    Split: {}\n'.format(tmp)
    fmt_str += '    Root Location: {}\n'.format(self.root)
    return fmt_str
