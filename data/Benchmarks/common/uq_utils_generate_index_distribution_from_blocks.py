def generate_index_distribution_from_blocks(numTrain, numTest,
    numValidation, params):
    """Generates a vector of indices to partition the data for training.
    NO CHECKING IS DONE: it is assumed that the data could be partitioned
    in the specified block quantities and that the block quantities describe a
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
        (uq_train_bks, uq_valid_bks, uq_test_bks)

    Return
    ----------
    indexTrain : int numpy array
        Indices for data in training
    indexValidation : int numpy array
        Indices for data in validation (if any)
    indexTest : int numpy array
        Indices for data in testing (if merging)
    """
    numBlocksTrain = params['uq_train_bks']
    numBlocksValidation = params['uq_valid_bks']
    numBlocksTest = params['uq_test_bks']
    numBlocksTotal = numBlocksTrain + numBlocksValidation + numBlocksTest
    if numBlocksTest > 0:
        numData = numTrain + numValidation + numTest
    else:
        numData = numTrain + numValidation
    blockSize = (numData + numBlocksTotal // 2) // numBlocksTotal
    remainder = numData - blockSize * numBlocksTotal
    if remainder != 0:
        print(
            'Warning ! Requested partition does not distribute data evenly between blocks. Testing (if specified) or Validation (if specified) will use different block size.'
            )
    sizeTraining = numBlocksTrain * blockSize
    sizeValidation = numBlocksValidation * blockSize
    Folds = np.arange(numData)
    np.random.shuffle(Folds)
    indexTrain = Folds[:sizeTraining]
    indexValidation = None
    if numBlocksValidation > 0:
        indexValidation = Folds[sizeTraining:sizeTraining + sizeValidation]
    indexTest = None
    if numBlocksTest > 0:
        indexTest = Folds[sizeTraining + sizeValidation:]
    return indexTrain, indexValidation, indexTest
