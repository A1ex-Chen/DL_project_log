def generate_index_distribution_from_block_list(numTrain, numTest,
    numValidation, params):
    """Generates a vector of indices to partition the data for training.
    NO CHECKING IS DONE: it is assumed that the data could be partitioned
    in the specified list of blocks and that the block indices describe a
    coherent partition.

    Parameters
    ----------
    numTrain : int
        Number of training data points
    numTest : int
        Number of testing data points
    numValidation : int
        Number of validation data points (may be zero)
    params : dictionary with parameters
        Contains the keywords that control the behavior of the function
        (uq_train_vec, uq_valid_vec, uq_test_vec)

    Return
    ----------
    indexTrain : int numpy array
        Indices for data in training
    indexValidation : int numpy array
        Indices for data in validation (if any)
    indexTest : int numpy array
        Indices for data in testing (if merging)
    """
    blocksTrain = params['uq_train_vec']
    blocksValidation = params['uq_valid_vec']
    blocksTest = params['uq_test_vec']
    numBlocksTrain = len(blocksTrain)
    numBlocksValidation = len(blocksValidation)
    numBlocksTest = len(blocksTest)
    numBlocksTotal = numBlocksTrain + numBlocksValidation + numBlocksTest
    if numBlocksTest > 0:
        numData = numTrain + numValidation + numTest
    else:
        numData = numTrain + numValidation
    blockSize = (numData + numBlocksTotal // 2) // numBlocksTotal
    remainder = numData - blockSize * numBlocksTotal
    if remainder != 0:
        print(
            'Warning ! Requested partition does not distribute data evenly between blocks. Last block will have different size.'
            )
    if remainder < 0:
        remainder = 0
    maxSizeTrain = blockSize * numBlocksTrain + remainder
    indexTrain = fill_array(blocksTrain, maxSizeTrain, numData,
        numBlocksTotal, blockSize)
    indexValidation = None
    if numBlocksValidation > 0:
        maxSizeValidation = blockSize * numBlocksValidation + remainder
        indexValidation = fill_array(blocksValidation, maxSizeValidation,
            numData, numBlocksTotal, blockSize)
    indexTest = None
    if numBlocksTest > 0:
        maxSizeTest = blockSize * numBlocksTest + remainder
        indexTest = fill_array(blocksTest, maxSizeTest, numData,
            numBlocksTotal, blockSize)
    return indexTrain, indexValidation, indexTest
