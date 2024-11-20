def generate_index_distribution_from_fraction(numTrain, numTest,
    numValidation, params):
    """Generates a vector of indices to partition the data for training.
    It checks that the fractions provided are (0, 1) and add up to 1.

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
        (uq_train_fr, uq_valid_fr, uq_test_fr)

    Return
    ----------
    indexTrain : int numpy array
        Indices for data in training
    indexValidation : int numpy array
        Indices for data in validation (if any)
    indexTest : int numpy array
        Indices for data in testing (if merging)
    """
    tol = 1e-07
    fractionTrain = params['uq_train_fr']
    fractionValidation = params['uq_valid_fr']
    fractionTest = params['uq_test_fr']
    if fractionTrain < 0.0 or fractionTrain > 1.0:
        raise ValueError('uq_train_fr is not in (0, 1) range. uq_train_fr: ',
            fractionTrain)
    if fractionValidation < 0.0 or fractionValidation > 1.0:
        raise ValueError('uq_valid_fr is not in (0, 1) range. uq_valid_fr: ',
            fractionValidation)
    if fractionTest < 0.0 or fractionTest > 1.0:
        raise ValueError('uq_test_fr is not in (0, 1) range. uq_test_fr: ',
            fractionTest)
    fractionSum = fractionTrain + fractionValidation + fractionTest
    if abs(fractionSum - 1.0) > tol:
        raise ValueError(
            'Specified UQ fractions (uq_train_fr, uq_valid_fr, uq_test_fr) do not add up to 1. No cross-validation partition is computed ! sum:'
            , fractionSum)
    if fractionTest > 0:
        numData = numTrain + numValidation + numTest
    else:
        numData = numTrain + numValidation
    sizeTraining = int(np.round(numData * fractionTrain))
    sizeValidation = int(np.round(numData * fractionValidation))
    Folds = np.arange(numData)
    np.random.shuffle(Folds)
    indexTrain = Folds[:sizeTraining]
    indexValidation = None
    if fractionValidation > 0:
        indexValidation = Folds[sizeTraining:sizeTraining + sizeValidation]
    indexTest = None
    if fractionTest > 0:
        indexTest = Folds[sizeTraining + sizeValidation:]
    return indexTrain, indexValidation, indexTest
