def bar(self):
    cell = self.cell
    X = self.inp
    H = self.hid
    B = self.b
    t = self.type
    subseqId = self.sub
    direc = self.dir
    name = self.name
    grid = self.grid
    multiple = self.multiple
    if direc == 'fprop':
        subseqId = subseqId % 3
        if subseqId == 0:
            self.gemm = 'layer'
            self.m = multiple * H
            self.n = B
            self.k = X
        elif subseqId == 1:
            self.gemm = 'recur'
            self.m = multiple * H
            self.n = B
            self.k = H
        else:
            layerGemmElems = multiple * H * B
            recurGemmElems = multiple * H * B
            cElems = H * B
            hElems = H * B
            totElems = layerGemmElems + recurGemmElems + 2 * cElems + hElems
            self.elems = totElems
    elif 'gemm' in name and hasTileSize(name):
        tileX, tileY = ctaTile(name)
        grid = grid.split(',')
        gridX, gridY, gridZ = map(lambda x: int(x), grid)
        gemmM = tileX * gridX
        gemmN = tileY * gridY
        if name[-3:] == '_nn':
            if gemmM == H:
                gemmN = B
                gemmK = multiple * H
                self.gemm = 'recur'
                self.m = gemmM
                self.n = gemmN
                self.k = gemmK
            elif gemmM == X:
                gemmK = multiple * H
                self.gemm = 'layer'
                self.m = gemmM
                self.n = gemmN
                self.k = gemmK
            else:
                pass
        elif name[-3:] == '_nt':
            if gemmM == H:
                assert gemmN == multiple * H
                gemmK = B
                self.gemm = 'recur'
                self.m = gemmM
                self.n = gemmN
                self.k = gemmK
            elif gemmM == X:
                assert gemmN == multiple * H
                gemmK = B
                self.gemm = 'layer'
                self.m = gemmM
                self.n = gemmN
                self.k = gemmK
            else:
                pass
        else:
            pass
    else:
        pass
    return
