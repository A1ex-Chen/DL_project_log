def getProfileStart(self):
    """
		Get the profile start time
		"""
    profStart = sys.maxsize
    for table in [self.driverT, self.runtimeT, self.kernelT, self.markerT]:
        colname = 'timestamp' if table is self.markerT else 'start'
        cmd = 'select {} from {} ORDER BY {} ASC LIMIT 1'.format(colname,
            table, colname)
        result = self.db.select(cmd)
        assert len(result) <= 1
        if len(result) == 1:
            assert colname in result[0]
            t = result[0][colname]
            if t < profStart:
                profStart = t
    assert profStart < sys.maxsize
    return profStart
