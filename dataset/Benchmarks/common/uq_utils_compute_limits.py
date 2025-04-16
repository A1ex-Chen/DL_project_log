def compute_limits(numdata, numblocks, blocksize, blockn):
    """Generates the limit of indices corresponding to a
    specific block. It takes into account the non-exact
    divisibility of numdata into numblocks letting the
    last block to take the extra chunk.

    Parameters
    ----------
    numdata : int
        Total number of data points to distribute
    numblocks : int
        Total number of blocks to distribute into
    blocksize : int
        Size of data per block
    blockn : int
        Index of block, from 0 to numblocks-1

    Return
    ----------
    start : int
        Position to start assigning indices
    end : int
        One beyond position to stop assigning indices
    """
    start = blockn * blocksize
    end = start + blocksize
    if blockn == numblocks - 1:
        end = numdata
    return start, end
