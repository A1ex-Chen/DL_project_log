def fillsquares():
    """For each square of points stepsize apart,
        calculate middle value as mean of points + wibble"""
    cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
    squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
    squareaccum += np.roll(squareaccum, shift=-1, axis=1)
    maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize
        ] = wibbledmean(squareaccum)
