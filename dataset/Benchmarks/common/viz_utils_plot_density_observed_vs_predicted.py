def plot_density_observed_vs_predicted(Ytest, Ypred, pred_name=None,
    figprefix=None):
    """Functionality to plot a 2D histogram of the distribution of observed (ground truth)
       values vs. predicted values. The plot generated is stored in a png file.

    Parameters
    ----------
    Ytest : numpy array
      Array with (true) observed values
    Ypred : numpy array
      Array with predicted values.
    pred_name : string
      Name of data colum or quantity predicted (e.g. growth, AUC, etc.)
    figprefix : string
      String to prefix the filename to store the figure generated.
      A '_density_predictions.png' string will be appended to the
      figprefix given.
    """
    xbins = 51
    plt.figure(figsize=(24, 18))
    ax = plt.gca()
    plt.rc('xtick', labelsize=16)
    ax.plot([Ytest.min(), Ytest.max()], [Ytest.min(), Ytest.max()], 'r--',
        lw=4.0)
    plt.hist2d(Ytest, Ypred, bins=xbins, norm=LogNorm())
    cb = plt.colorbar()
    ax.set_xlabel('Observed ' + pred_name, fontsize=38, labelpad=15.0)
    ax.set_ylabel('Mean ' + pred_name + ' Predicted', fontsize=38, labelpad
        =15.0)
    ax.axis([Ytest.min() * 0.98, Ytest.max() * 1.02, Ytest.min() * 0.98, 
        Ytest.max() * 1.02])
    plt.setp(ax.get_xticklabels(), fontsize=32)
    plt.setp(ax.get_yticklabels(), fontsize=32)
    cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=28)
    plt.grid(True)
    plt.savefig(figprefix + '_density_predictions.png', bbox_inches='tight')
    plt.close()
    print('Generated plot: ', figprefix + '_density_predictions.png')
