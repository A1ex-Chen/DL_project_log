def getMarkerInfo(self, objId, startTime, endTime):
    """
		This function first finds all NVTX markers encapsulating
		a runtime / driver kernel launch.
		It then splits the markers into many lists.
			layerMarkers : User added NVTX markers
			traceMarkers : Call trace markers (inserted by pyprof)
			reprMarkers  : Markers containing the extra_repr() of a module (inserted by pyprof)
			pyprofMarkers: Markers containing args and kwargs (tensor shape, datatype etc.)
			seqMarkers   : Markers containing PyTorch internal sequence markers (inserted by PyTorch)
			altSeqMarkers: Markers inserted by PyTorch between two kernel launches. Needs better explanation.
			otherMarkers : Markers not in either of the above categories.

		We extract seqId from the seq and altSeq markers. The seqId is used in bprop.
		We also extract information from the layerMarkers.
		"""
    layerMarkers = []
    traceMarkers = []
    reprMarkers = []
    pyprofMarkers = []
    seqMarkers = []
    otherMarkers = []
    altSeqMarkers = []
    bprop = False

    def delete(objId, sTime):
        """
			Delete rows from the temporary SQL table which are no longer required.
			This speeds up future queries.
			"""
        margin = 0
        cmd = ('DELETE FROM marker WHERE objectId = "{}" AND endTime < {}'.
            format(objId, sTime - margin))
        self.db.execute(cmd)

    def getLayerName(mlist):
        """
			Get layer names from layer marker list.
			"""
        layers = []
        assert type(mlist) == list
        for m in mlist:
            assert 'layer:' in m
            l = m.split(':')[1]
            layers.append(l)
        return layers

    def getSeqId(mlist):
        """
			Get sequence ids from seq / alt seq marker list.
			"""
        ids = []
        assert type(mlist) == list
        for m in mlist:
            assert ', seq = ' in m
            seq = int(m.split('=')[1])
            ids.append(seq)
        ids = list(set(ids))
        ids.sort()
        return ids

    def seqcompare(elem):
        """
			Sorting function for sequence markers
			"""
        assert ', seq = ' in elem
        l = elem.split(' = ')
        return l[1] + l[0]

    def prune(mlist):
        """
			Remove markers with the same seqId and if the strings are similar.
			This function works on a sorted sequence.
			"""
        assert type(mlist) == list
        assert len(mlist)
        a = mlist[0:1]
        for i in range(1, len(mlist)):
            m = mlist[i]
            pm = mlist[i - 1]
            name, seq = m.split(',')
            pname, pseq = pm.split(',')
            similar = name in pname or pname in name
            if seq == pseq and similar:
                continue
            else:
                a.append(m)
        return a

    def filterTrace(mlist):
        """
			Filter trace markers to remove certain file names.
			"""
        assert type(mlist) == list
        if len(mlist) == 0:
            return mlist
        mlist = mlist[-1]
        mlist = eval(mlist)
        mlist = mlist['traceMarker']
        assert type(mlist) == list
        mlist = list(filter(lambda x: '/torch/nn/modules/' not in x, mlist))
        mlist = list(filter(lambda x: '/torch/nn/functional.py' not in x,
            mlist))
        mlist = list(filter(lambda x: '/torch/tensor.py' not in x, mlist))
        mlist = list(filter(lambda x: '/torch/autograd/__init__.py' not in
            x, mlist))
        mlist = list(filter(lambda x: '/torch/_jit_internal.py' not in x,
            mlist))
        mlist = list(filter(lambda x: '/pyprof/nvtx/nvmarker.py' not in x,
            mlist))
        mlist = list(filter(lambda x: '/apex/optimizers/' not in x, mlist))
        mlist = list(filter(lambda x: '/torch/_utils.py' not in x, mlist))
        mlist = list(filter(lambda x: '/torch/optim/' not in x, mlist))
        return mlist
    cmd = (
        'SELECT id,name from marker where \t\t\t\tobjectId = "{}" and \t\t\t\tstartTime < {} and \t\t\t\tendTime > {} \t\t\t\tORDER BY startTime ASC'
        .format(objId, startTime, endTime))
    result = self.db.select(cmd)
    for r in result:
        m = self.getString(r['name'])
        if m.find('CheckpointFunctionBackward') >= 0:
            continue
        if ('_backward, seq =' in m or 'Backward, seq =' in m or 
            'Backward0, seq =' in m):
            bprop = True
        if 'mod' in m and 'op' in m and 'args' in m and 'type' in m:
            pyprofMarkers.append(m)
        elif 'layer:' in m:
            layerMarkers.append(m)
        elif 'traceMarker' in m:
            traceMarkers.append(m)
        elif 'strRepr' in m:
            reprMarkers.append(m)
        elif ', seq = ' in m:
            seqMarkers.append(m)
        else:
            otherMarkers.append(m)
    if len(seqMarkers):
        seqMarkers = list(set(seqMarkers))
        seqMarkers.sort(key=seqcompare)
        seqMarkers = prune(seqMarkers)
    otherMarkers = list(set(otherMarkers))
    if len(result) and not bprop:
        loId = self.markerId
        hiId = result[-1]['id']
        self.markerId = hiId
        cmd = (
            'SELECT id,name from marker where objectId = "{}" and id > {} and id < {} ORDER BY startTime ASC'
            .format(objId, loId, hiId))
        result1 = self.db.select(cmd)
        for r in result1:
            m = self.getString(r['name'])
            if ', seq=' in m:
                altSeqMarkers.append(m)
        if len(altSeqMarkers):
            altSeqMarkers = list(set(altSeqMarkers))
            altSeqMarkers.sort(key=seqcompare)
            altSeqMarkers = prune(altSeqMarkers)
    delete(objId, startTime)
    return (layerMarkers, filterTrace(traceMarkers), reprMarkers,
        pyprofMarkers, seqMarkers, otherMarkers, altSeqMarkers, getSeqId(
        seqMarkers), getSeqId(altSeqMarkers), getLayerName(layerMarkers))
