def print(self):
    """
		Print kernel information. This is used by prof.py.
		"""
    a = lambda : None
    a.kShortName = self.kShortName
    a.kDuration = self.kDuration
    a.layer = self.layer
    a.trace = self.traceMarkers
    a.reprMarkers = self.reprMarkers
    a.marker = self.pyprofMarkers
    a.seqMarker = self.seqMarkers
    a.seqId = self.seqId
    a.subSeqId = self.subSeqId
    a.altSeqId = self.altSeqId
    a.dir = self.dir
    a.mod = self.mod
    a.op = self.op
    a.tid = self.tid
    a.device = self.device
    a.stream = self.stream
    a.grid = self.grid
    a.block = self.block
    a.kLongName = self.kLongName
    print(a.__dict__)
