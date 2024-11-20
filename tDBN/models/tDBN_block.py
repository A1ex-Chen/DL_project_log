def block(self, m, a, b, dimension=3, residual_blocks=False, leakiness=0):
    if residual_blocks:
        m.add(scn.ConcatTable().add(scn.Identity() if a == b else scn.
            NetworkInNetwork(a, b, False)).add(scn.Sequential().add(scn.
            BatchNormLeakyReLU(a, leakiness=leakiness)).add(scn.
            SubmanifoldConvolution(dimension, a, b, 3, False)).add(scn.
            BatchNormLeakyReLU(b, leakiness=leakiness)).add(scn.
            SubmanifoldConvolution(dimension, b, b, 3, False)))).add(scn.
            AddTable())
    else:
        m.add(scn.Sequential().add(scn.BatchNormLeakyReLU(a, leakiness=
            leakiness)).add(scn.SubmanifoldConvolution(dimension, a, b, 3, 
            False)))
