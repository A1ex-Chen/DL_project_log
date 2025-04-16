import sys

class NVVP(object):
	"""
	This class gets kernel information from the SQL (nvvp) database.
	"""

	driverT = "CUPTI_ACTIVITY_KIND_DRIVER"
	runtimeT = "CUPTI_ACTIVITY_KIND_RUNTIME"
	kernelT = "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL"
	markerT = "CUPTI_ACTIVITY_KIND_MARKER"
	stringT = "StringTable"













		#Find all encapsulating markers
		cmd = 'SELECT id,name from marker where \
				objectId = "{}" and \
				startTime < {} and \
				endTime > {} \
				ORDER BY startTime ASC'.format(objId, startTime, endTime)
		result = self.db.select(cmd)

		#Bin markers into different lists
		for r in result:
			m = self.getString(r['name'])

			#Hack: If its a known gradient checkpointing marker, ignore it.
			if m.find("CheckpointFunctionBackward") >= 0:
				continue

			if ("_backward, seq =" in m) or ("Backward, seq =" in m) or ("Backward0, seq =" in m):
				bprop = True

			if ("mod" in m) and ("op" in m) and ("args" in m) and ("type" in m):
				pyprofMarkers.append(m)
			elif ("layer:" in m):
				layerMarkers.append(m)
			elif ("traceMarker" in m):
				traceMarkers.append(m)
			elif ("strRepr" in m):
				reprMarkers.append(m)
			elif (", seq = " in m):
				seqMarkers.append(m)
			else:
				otherMarkers.append(m)

		#Remove duplicates, sort and prune seqMarkers
		if (len(seqMarkers)):
			seqMarkers = list(set(seqMarkers))
			seqMarkers.sort(key=seqcompare)
			seqMarkers = prune(seqMarkers)

		#Remove duplicates from otherMarkers
		otherMarkers = list(set(otherMarkers))

		#Get markers with seq id (inserted by PyTorch) from the previous kernel to the present kernel
		#Only for fprop kernels
		if (len(result) and not bprop):
			loId = self.markerId
			hiId = result[-1]['id']
			self.markerId = hiId
			
			#Get markers between loId and hiId
			cmd = 'SELECT id,name from marker where objectId = "{}" and id > {} and id < {} ORDER BY startTime ASC'.format(objId, loId, hiId)
			result1 = self.db.select(cmd)

			for r in result1:
				m = self.getString(r['name'])
				#Get only markers with seq id
				if (", seq=" in m):
					altSeqMarkers.append(m)

			#Remove duplicates, sort and prune altSeqMarkers
			if (len(altSeqMarkers)):
				altSeqMarkers = list(set(altSeqMarkers))
				altSeqMarkers.sort(key=seqcompare)
				altSeqMarkers = prune(altSeqMarkers)

		delete(objId, startTime)

		return layerMarkers, filterTrace(traceMarkers), reprMarkers, pyprofMarkers, seqMarkers, otherMarkers, altSeqMarkers, getSeqId(seqMarkers), getSeqId(altSeqMarkers), getLayerName(layerMarkers)