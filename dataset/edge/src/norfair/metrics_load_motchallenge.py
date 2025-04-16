def load_motchallenge(matrix_data, min_confidence=-1):
    """Load MOT challenge data.

    This is a modification of the function load_motchallenge from the py-motmetrics library, defined in io.py
    In this version, the pandas dataframe is generated from a numpy array (matrix_data) instead of a text file.

    Params
    ------
    matrix_data : array  of float that has [frame, id, X, Y, width, height, conf, cassId, visibility] in each row, for each prediction on a particular video

    min_confidence : float
        Rows with confidence less than this threshold are removed.
        Defaults to -1. You should set this to 1 when loading
        ground truth MOTChallenge data, so that invalid rectangles in
        the ground truth are not considered during matching.

    Returns
    ------
    df : pandas.DataFrame
        The returned dataframe has the following columns
            'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility'
        The dataframe is indexed by ('FrameId', 'Id')
    """
    df = pd.DataFrame(data=matrix_data, columns=['FrameId', 'Id', 'X', 'Y',
        'Width', 'Height', 'Confidence', 'ClassId', 'Visibility', 'unused'])
    df = df.set_index(['FrameId', 'Id'])
    df[['X', 'Y']] -= 1, 1
    del df['unused']
    return df[df['Confidence'] >= min_confidence]
