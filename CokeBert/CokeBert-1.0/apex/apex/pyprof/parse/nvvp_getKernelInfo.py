def getKernelInfo(self):
    """
		Get GPU kernel info
		"""
    cmd = (
        'select name,correlationId,start,end,deviceId,streamId,gridX,gridY,gridZ,blockX,blockY,blockZ from {}'
        .format(self.kernelT))
    result = self.db.select(cmd)
    return result
