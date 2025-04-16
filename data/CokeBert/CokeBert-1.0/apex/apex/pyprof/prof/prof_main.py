def main():
    cmdArgs = parseArgs()
    output = Output(cmdArgs)
    output.header()
    idx = -1
    for line in cmdArgs.file:
        idx += 1
        kernel = eval(line)
        assert kernel
        kernels.append(kernel)
        k = kernel
        d = Data(k)
        mod = k['mod']
        op = k['op']
        flops = 0
        params = {'na': 'na'}
        tc = 'na'
        bytes = 0
        if d.dir == 'bprop':
            d.seqMarker = k['seqMarker']
            seq = k['seqId']
            if len(seq) > 1:
                pass
            seq = k['seqId'][:1]
            assert len(seq) == 1, seq
            assert len(d.seqMarker) > 0
            if len(d.argMarker) == 0:
                index = findFpropKernel(seq[0])
                if index >= 0:
                    d.argMarker = kernels[index]['marker']
                    d.modMarker = kernels[index]['reprMarkers']
                    mod = kernels[index]['mod']
                    op = kernels[index]['op']
                    d.layer = kernels[index]['layer']
                    d.trace = kernels[index]['trace']
        if len(d.argMarker) and Utility.hasNVTX(d.argMarker[0]):
            xx = foo(mod, op, d)
            bytes = xx.bytes()
            flops = xx.flops()
            op = xx.op()
            params = xx.params()
            tc = xx.tc()
        if type(op) is list:
            if len(op):
                op = op[0]
            else:
                op = ''
        if type(mod) is list:
            if len(mod):
                mod = mod[0]
            else:
                mod = ''
        d.index = idx + 1
        d.setParams(params)
        d.tc = tc
        d.flops = flops
        d.bytes = bytes
        d.mod = mod
        d.op = op
        output.data(d)
