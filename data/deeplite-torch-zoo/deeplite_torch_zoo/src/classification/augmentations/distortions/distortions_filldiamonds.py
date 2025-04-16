def filldiamonds():
    """For each diamond of points stepsize apart,
        calculate middle value as mean of points + wibble"""
    mapsize = maparray.shape[0]
    drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize
        :stepsize]
    ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
    ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
    lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
    ltsum = ldrsum + lulsum
    maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(
        ltsum)
    tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
    tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
    ttsum = tdrsum + tulsum
    maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(
        ttsum)
