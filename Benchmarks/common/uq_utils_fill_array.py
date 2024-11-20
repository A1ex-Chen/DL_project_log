def fill_array(blocklist, maxsize, numdata, numblocks, blocksize):
    """Fills a new array of integers with the indices corresponding
    to the specified block structure.

    Parameters
    ----------
    blocklist : list
        List of integers describen the block indices that
        go into the array
    maxsize : int
        Maximum possible length for the partition (the size of the
        common block size plus the remainder, if any).
    numdata : int
        Total number of data points to distribute
    numblocks : int
        Total number of blocks to distribute into
    blocksize : int
        Size of data per block

    Return
    ----------
    indexArray : int numpy array
        Indices for specific data partition. Resizes the array
        to the correct length.
    """
    indexArray = np.zeros(maxsize, np.int)
    offset = 0
    for i in blocklist:
        start, end = compute_limits(numdata, numblocks, blocksize, i)
        length = end - start
        indexArray[offset:offset + length] = np.arange(start, end)
        offset += length
    return indexArray[:offset]
