def plot_calibration_interpolation(mean_sigma, error, splineobj1,
    splineobj2, method='', figprefix=None, steps=False):
    """Functionality to plot empirical calibration curves
       estimated by interpolation of the computed
       standard deviations and errors. Since the estimations
       are very noisy, two levels of smoothing are used. Both
       can be plotted independently, if requested.
       The plot(s) generated is(are) stored in png file(s).

    Parameters
    ----------
    mean_sigma : numpy array
      Array with the mean standard deviations computed in inference.
    error : numpy array
      Array with the errors computed from the means predicted in inference.
    splineobj1 : scipy.interpolate python object
      A python object from scipy.interpolate that computes a
      cubic Hermite spline (PchipInterpolator) to express
      the interpolation after the first smoothing. This
      spline is a partial result generated during the empirical
      calibration procedure.
    splineobj2 : scipy.interpolate python object
      A python object from scipy.interpolate that computes a
      cubic Hermite spline (PchipInterpolator) to express
      the mapping from standard deviation to error. This
      spline is generated for interpolating the predictions
      after a process of smoothing-interpolation-smoothing
      computed during the empirical calibration procedure.
    method : string
      Method used to comput the standard deviations (i.e. dropout,
      heteroscedastic, etc.).
    figprefix : string
      String to prefix the filename to store the figure generated.
      A '_empirical_calibration_interpolation.png' string will be appended to
      the figprefix given.
    steps : boolean
      Besides the complete empirical calibration (including the interpolating
      spline), also generates partial plots with only the spline of
      the interpolating spline after the first smoothing level (smooth1).
    """
    xmax = np.max(mean_sigma)
    xmin = np.min(mean_sigma)
    xp23 = np.linspace(xmin, xmax, 200)
    yp23 = splineobj2(xp23)
    if steps:
        yp23_1 = splineobj1(xp23)
        fig = plt.figure(figsize=(24, 18))
        ax = plt.gca()
        ax.plot(mean_sigma, error, 'kx')
        ax.plot(xp23, yp23_1, 'gx', ms=20)
        plt.legend(['True', 'Cubic Spline'], fontsize=28)
        plt.xlabel('Standard Deviation Predicted (' + method + ')',
            fontsize=38.0)
        plt.ylabel('Error: ABS Observed - Mean Predicted', fontsize=38.0)
        plt.title('Calibration (by Interpolation)', fontsize=40)
        plt.setp(ax.get_xticklabels(), fontsize=32)
        plt.setp(ax.get_yticklabels(), fontsize=32)
        plt.grid()
        fig.tight_layout()
        plt.savefig(figprefix + '_empirical_calibration_interp_smooth1.png',
            bbox_inches='tight')
        plt.close()
        print('Generated plot: ', figprefix +
            '_empirical_calibration_interp_smooth1.png')
    fig = plt.figure(figsize=(24, 18))
    ax = plt.gca()
    ax.plot(mean_sigma, error, 'kx')
    ax.plot(xp23, yp23, 'rx', ms=20)
    plt.legend(['True', 'Cubic Spline'], fontsize=28)
    plt.xlabel('Standard Deviation Predicted (' + method + ')', fontsize=38.0)
    plt.ylabel('Error: ABS Observed - Mean Predicted', fontsize=38.0)
    plt.title('Calibration (by Interpolation)', fontsize=40)
    plt.setp(ax.get_xticklabels(), fontsize=32)
    plt.setp(ax.get_yticklabels(), fontsize=32)
    plt.grid()
    fig.tight_layout()
    plt.savefig(figprefix + '_empirical_calibration_interpolation.png',
        bbox_inches='tight')
    plt.close()
    print('Generated plot: ', figprefix +
        '_empirical_calibration_interpolation.png')
