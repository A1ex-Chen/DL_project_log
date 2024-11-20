def findFpropKernel(seq):
    for idx in reversed(range(len(kernels))):
        k = kernels[idx]
        if seq in k['seqId'] and k['dir'] == 'fprop':
            return idx
    for idx in reversed(range(len(kernels))):
        k = kernels[idx]
        if seq in k['altSeqId'] and k['dir'] == 'fprop':
            return idx
    return -1
