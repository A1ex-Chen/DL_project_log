def plot_calibrated_std(y_test, y_pred, std_calibrated, thresC, pred_name=
    None, figprefix=None):
    """Functionality to plot values in testing set after calibration. An estimation of the lower-confidence samples is made. The plot generated is stored in a png file.

    Parameters
    ----------
    y_test : numpy array
      Array with (true) observed values.
    y_pred : numpy array
      Array with predicted values.
    std_calibrated : numpy array
      Array with standard deviation values after calibration.
    thresC : float
      Threshold to label low confidence predictions (low
      confidence predictions are the ones with std > thresC).
    pred_name : string
      Name of data colum or quantity predicted (e.g. growth, AUC, etc.).
    figprefix : string
      String to prefix the filename to store the figure generated.
      A '_calibrated.png' string will be appended to the
      figprefix given.
    """
    N = y_test.shape[0]
    index = np.argsort(y_pred)
    x = np.array(range(N))
    indexC = std_calibrated > thresC
    alphafill = 0.5
    if N > 2000:
        alphafill = 0.7
    scale = 120
    fig = plt.figure(figsize=(24, 18))
    ax = plt.gca()
    ax.scatter(x, y_test[index], color='red', s=scale, alpha=0.5)
    plt.fill_between(x, y_pred[index] - 1.28 * std_calibrated[index], 
        y_pred[index] + 1.28 * std_calibrated[index], color='gray', alpha=
        alphafill)
    plt.scatter(x, y_pred[index], color='orange', s=scale)
    plt.scatter(x[indexC], y_test[indexC], color='green', s=scale, alpha=0.5)
    plt.legend(['True', '1.28 Std', 'Pred', 'Low conf'], fontsize=28)
    plt.xlabel('Index', fontsize=38.0)
    plt.ylabel(pred_name + ' Predicted', fontsize=38.0)
    plt.title('Calibrated Standard Deviation', fontsize=40)
    plt.setp(ax.get_xticklabels(), fontsize=32)
    plt.setp(ax.get_yticklabels(), fontsize=32)
    plt.grid()
    fig.tight_layout()
    plt.savefig(figprefix + '_calibrated.png', bbox_inches='tight')
    plt.close()
    print('Generated plot: ', figprefix + '_calibrated.png')
