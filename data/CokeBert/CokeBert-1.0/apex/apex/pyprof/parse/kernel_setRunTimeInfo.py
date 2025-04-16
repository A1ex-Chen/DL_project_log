def setRunTimeInfo(self, info):
    start, end, pid, tid = info
    self.rStartTime = start
    self.rEndTime = end
    self.rDuration = end - start
    self.pid = pid
    self.tid = tid
    self.objId = encode_object_id(pid, tid)
