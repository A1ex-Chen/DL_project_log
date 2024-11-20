def main():
    args = parseArgs()
    db = DB(args.file)
    nvvp = NVVP(db)
    kInfo = nvvp.getKernelInfo()
    if len(kInfo) == 0:
        print('Found 0 kernels. Exiting.', file=sys.stderr)
        db.close()
        sys.exit(0)
    else:
        print('Found {} kernels. Getting info for each kernel.'.format(len(
            kInfo)), file=sys.stderr)
    nvvp.createMarkerTable()
    prevSeqId = -1
    prevSubSeqId = -1
    prevOp = 'na'
    Kernel.profStart = nvvp.getProfileStart()
    for i in tqdm(range(len(kInfo)), ascii=True):
        info = kInfo[i]
        k = Kernel()
        k.setKernelInfo(info)
        name = nvvp.getString(k.kNameId)
        k.setKernelName(name)
        info = nvvp.getCPUInfo(k.corrId)
        k.setRunTimeInfo(info)
        info = nvvp.getMarkerInfo(k.objId, k.rStartTime, k.rEndTime)
        k.setMarkerInfo(info)
        if any(seq != 0 for seq in k.seqId) and 0 in k.seqId:
            k.seqId.remove(0)
        k.setDirection()
        k.setOp()
        if len(k.seqId):
            assert k.dir in ['fprop', 'bprop']
            if k.dir == 'fprop':
                inc = k.seqId[-1] > prevSeqId
                if inc:
                    currSeqId = [x for x in k.seqId if x > prevSeqId][0]
                else:
                    currSeqId = prevSeqId
            else:
                currSeqId = k.seqId[0]
            if currSeqId == prevSeqId and k.op == prevOp or k.op[0
                ] == 'forward' and k.op == prevOp and k.mod[0] in ['LSTMCell',
                'GRUCell', 'RNNCell']:
                k.subSeqId = prevSubSeqId + 1
            prevSeqId = currSeqId
            prevSubSeqId = k.subSeqId
            prevOp = k.op
            for s in k.seqId:
                if s != currSeqId:
                    k.seqId.remove(s)
                    k.altSeqId.append(s)
            for s in k.altSeqId:
                if s == currSeqId:
                    k.altSeqId.remove(s)
            k.altSeqId = list(set(k.altSeqId))
            if len(k.altSeqId):
                k.altSeqId.sort()
        k.print()
    db.close()
