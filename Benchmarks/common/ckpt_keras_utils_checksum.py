def checksum(self, dir_work):
    """
        Simple checksum dispatch
        dir_work: A PosixPath
        """
    if self.checksum_enabled:
        self.cksum_model = checksum_file(self.logger, dir_work / 'model.h5')
    else:
        self.cksum_model = '__DISABLED__'
