def compute_empirical_calibration_interpolation(pSigma_cal, pPred_cal,
    true_cal, cv=10):
    """Use the arrays provided to estimate an empirical mapping
    between standard deviation and absolute value of error,
    both of which have been observed during inference. Since
    most of the times the prediction statistics are very noisy,
    two smoothing steps (based on scipy's savgol filter) are performed.
    Cubic Hermite splines (PchipInterpolator) are constructed for
    interpolation. This type of splines preserves the monotonicity
    in the interpolation data and does not overshoot if the data is
    not smooth. The overall process of constructing a spline
    to express the mapping from standard deviation to error is
    composed of smoothing-interpolation-smoothing-interpolation.

    Parameters
    ----------
    pSigma_cal : numpy array
        Part of the standard deviations array to use for calibration.
    pPred_cal : numpy array
        Part of the predictions array to use for calibration.
    true_cal : numpy array
        Part of the true (observed) values array to use for calibration.
    cv : int
        Number of cross validations folds to run to determine a 'good'
        fit.

    Return
    ----------
    splineobj_best : scipy.interpolate python object
        A python object from scipy.interpolate that computes a
        cubic Hermite splines (PchipInterpolator) constructed
        to express the mapping from standard deviation to error after a
        'drastic' smoothing of the predictions. A 'good' fit is
        determined by taking the spline for the fold that produces
        the smaller mean absolute error in testing data (not used
        for the smoothing / interpolation).
    splineobj2 : scipy.interpolate python object
        A python object from scipy.interpolate that computes a
        cubic Hermite splines (PchipInterpolator) constructed
        to express the mapping from standard deviation to error. This
        spline is generated for interpolating the samples generated
        after the smoothing of the first interpolation spline (i.e.
        splineobj_best).
    """
    xs3 = pSigma_cal
    z3 = np.abs(true_cal - pPred_cal)
    test_split = 1.0 / cv
    xmin = np.min(pSigma_cal)
    xmax = np.max(pSigma_cal)
    warnings.filterwarnings('ignore')
    print('--------------------------------------------')
    print('Using CV for selecting calibration smoothing')
    print('--------------------------------------------')
    min_error = np.inf
    for cv_ in range(cv):
        X_train, X_test, y_train, y_test = train_test_split(xs3, z3,
            test_size=test_split, shuffle=True)
        xindsort = np.argsort(X_train)
        z3smooth = signal.savgol_filter(y_train[xindsort], 21, 1, mode=
            'nearest')
        splineobj = interpolate.PchipInterpolator(X_train[xindsort],
            z3smooth, extrapolate=True)
        ytest = splineobj(X_test)
        mae = mean_absolute_error(y_test, ytest)
        print('MAE: ', mae)
        if mae < min_error:
            min_error = mae
            splineobj_best = splineobj
    xp23 = np.linspace(xmin, xmax, 200)
    yp23 = splineobj_best(xp23)
    yp23smooth = signal.savgol_filter(yp23, 15, 1, mode='nearest')
    splineobj2 = interpolate.PchipInterpolator(xp23, yp23smooth,
        extrapolate=True)
    return splineobj_best, splineobj2
