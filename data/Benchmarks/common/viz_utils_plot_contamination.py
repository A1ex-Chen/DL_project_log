def plot_contamination(y_true, y_pred, sigma, T=None, thresC=0.1, pred_name
    =None, figprefix=None):
    """Functionality to plot results for the contamination model.
       This includes the latent variables T if they are given (i.e.
       if the results provided correspond to training results). Global
       parameters for the normal distribution are used for shading 80%
       confidence interval.
       If results for training (i.e. T available), samples determined to
       be outliers (i.e. samples whose probability of membership to the
       heavy tailed distribution (Cauchy) is greater than the threshold
       given) are highlighted.
       The plot(s) generated is(are) stored in a png file.

    Parameters
    ----------
    y_true : numpy array
      Array with observed values.
    y_pred : numpy array
      Array with predicted values.
    sigma : float
      Standard deviation of the normal distribution.
    T : numpy array
      Array with latent variables (i.e. membership to normal and heavy-tailed
      distributions). If in testing T is not available (i.e. None)
    thresC : float
      Threshold to label outliers (outliers are the ones
      with probability of membership to heavy-tailed distribution,
      i.e. T[:,1] > thresC).
    pred_name : string
      Name of data colum or quantity predicted (e.g. growth, AUC, etc.).
    figprefix : string
      String to prefix the filename to store the figures generated.
      A '_contamination.png' string will be appended to the
      figprefix given.
    """
    N = y_true.shape[0]
    index = np.argsort(y_pred)
    x = np.array(range(N))
    if T is not None:
        indexG = T[:, 0] > 1.0 - thresC
        indexC = T[:, 1] > thresC
        ss = sigma * indexG
        prefig = '_outTrain'
    else:
        ss = sigma
        prefig = '_outTest'
    auxGh = y_pred + 1.28 * ss
    auxGl = y_pred - 1.28 * ss
    scale = 120
    fig = plt.figure(figsize=(24, 18))
    ax = plt.gca()
    ax.scatter(x, y_true[index], color='red', s=scale)
    if T is not None:
        plt.scatter(x[indexC], y_true[indexC], color='green', s=scale)
    plt.scatter(x, y_pred[index], color='orange', s=scale)
    plt.fill_between(x, auxGl[index], auxGh[index], color='gray', alpha=0.5)
    if T is not None:
        plt.legend(['True', 'Outlier', 'Pred', '1.28 Std'], fontsize=28)
    else:
        plt.legend(['True', 'Pred', '1.28 Std'], fontsize=28)
    plt.xlabel('Index', fontsize=38.0)
    plt.ylabel(pred_name + ' Predicted', fontsize=38.0)
    plt.title('Contamination Results', fontsize=40)
    plt.setp(ax.get_xticklabels(), fontsize=32)
    plt.setp(ax.get_yticklabels(), fontsize=32)
    plt.grid()
    fig.tight_layout()
    plt.savefig(figprefix + prefig + '_contamination.png', bbox_inches='tight')
    plt.close()
    print('Generated plot: ', figprefix + prefig + '_contamination.png')
    if T is not None:
        error = np.abs(y_true - y_pred)
        fig = plt.figure(figsize=(24, 18))
        ax = plt.gca()
        ax.scatter(error, T[:, 0], color='blue', s=scale)
        ax.scatter(error, T[:, 1], color='orange', s=scale)
        plt.legend(['Normal', 'Heavy-Tailed'], fontsize=28)
        plt.xlabel('ABS Error', fontsize=38.0)
        plt.ylabel('Membership Probability', fontsize=38.0)
        plt.title('Contamination: Latent Variables', fontsize=40)
        plt.setp(ax.get_xticklabels(), fontsize=32)
        plt.setp(ax.get_yticklabels(), fontsize=32)
        plt.grid()
        fig.tight_layout()
        plt.savefig(figprefix + '_T_contamination.png', bbox_inches='tight')
        plt.close()
        print('Generated plot: ', figprefix + '_T_contamination.png')
