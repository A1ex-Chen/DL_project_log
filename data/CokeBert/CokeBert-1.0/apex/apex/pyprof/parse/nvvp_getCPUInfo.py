def getCPUInfo(self, corrId):
    """
		Given the correlation id, get CPU start, end, thread id, process id.
		The information can be in the runtime table or the driver table.
		"""
    cmd = ('select start,end,processId,threadId from {} where correlationId={}'
        .format(self.runtimeT, corrId))
    result = self.db.select(cmd)
    assert len(result) <= 1
    if len(result) == 0:
        cmd = (
            'select start,end,processId,threadId from {} where correlationId={}'
            .format(self.driverT, corrId))
        result = self.db.select(cmd)
    assert len(result) == 1
    info = result[0]
    start = info['start']
    end = info['end']
    pid = info['processId']
    tid = info['threadId']
    tid = tid & 4294967295
    assert end > start
    return [start, end, pid, tid]
