@property
def raw_folder(self):
    return os.path.join(self.root, self.__class__.__name__, 'raw')
