import cxxfilt, struct, binascii

#Helper functions




class Kernel(object):
	"""
	This class stores information about a kernel.
	"""

	kernels = []
	profStart = 0









		#Check pyprof markers
		for m in self.pyprofMarkers:
			assert ("mod" in m) and ("op" in m) and ("args" in m)
			t = eval(m)
			self.op.append(t['op'])
			self.mod.append(t['mod'])

		if len(self.op):
			return

		#Check bprop kernel markers
		for m in self.seqMarkers:
			if ("backward, seq = " in m) or ("Backward, seq = " in m):
				op = m.split(",")[0]
				op = sanitize(op)
				self.op.append(op)
				self.mod.append('na')

		if len(self.op):
			return

		#Check markers with "seq = "
		for m in self.seqMarkers:
			if ", seq = " in m:
				op = m.split(",")[0]
				self.op.append(op)
				self.mod.append('na')

		if len(self.op):
			return

		#If nothing else
		if len(self.otherMarkers):
			self.op.append(self.otherMarkers[0])
		self.mod.append('na')

	def print(self):
		"""
		Print kernel information. This is used by prof.py.
		"""

		a = lambda: None
		a.kShortName = self.kShortName
		a.kDuration = self.kDuration
		#a.layerMarkers = self.layerMarkers
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