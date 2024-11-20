@property
def processed_folder(self):
    return os.path.join(self.root, self.__class__.__name__, 'processed')
