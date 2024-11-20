def generate_index_distribution(numTrain, numTest, numValidation, params):
    """Generates a vector of indices to partition the data for training.
    NO CHECKING IS DONE: it is assumed that the data could be partitioned
    in the specified blocks and that the block indices describe a coherent
    partition.

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
        (uq_train_fr, uq_valid_fr, uq_test_fr for fraction specification,
        uq_train_vec, uq_valid_vec, uq_test_vec for block list specification, and
        uq_train_bks, uq_valid_bks, uq_test_bks for block number specification)

    Return
    ----------
    indexTrain : int numpy array
        Indices for data in training
    indexValidation : int numpy array
        Indices for data in validation (if any)
    indexTest : int numpy array
        Indices for data in testing (if merging)
    """
    if all(k in params for k in ('uq_train_fr', 'uq_valid_fr', 'uq_test_fr')):
        print('Computing UQ cross-validation - Distributing by FRACTION')
        return generate_index_distribution_from_fraction(numTrain, numTest,
            numValidation, params)
    elif all(k in params for k in ('uq_train_vec', 'uq_valid_vec',
        'uq_test_vec')):
        print('Computing UQ cross-validation - Distributing by BLOCK LIST')
        return generate_index_distribution_from_block_list(numTrain,
            numTest, numValidation, params)
    elif all(k in params for k in ('uq_train_bks', 'uq_valid_bks',
        'uq_test_bks')):
        print('Computing UQ cross-validation - Distributing by BLOCK NUMBER')
        return generate_index_distribution_from_blocks(numTrain, numTest,
            numValidation, params)
    else:
        print(
            'ERROR !! No consistent UQ parameter specification found !! ... exiting '
            )
        raise KeyError(
            "No valid triplet of ('uq_train_*', 'uq_valid_*', 'uq_test_*') found. (* is any of fr, vec or bks)"
            )
