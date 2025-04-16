def plot_2d_density_sigma_vs_error(sigma, yerror, method=None, figprefix=None):
    """Functionality to plot a 2D histogram of the distribution of
       the standard deviations computed for the predictions vs. the
       computed errors (i.e. values of observed - predicted).
       The plot generated is stored in a png file.

    Parameters
    ----------
    sigma : numpy array
      Array with standard deviations computed.
    yerror : numpy array
      Array with errors computed (observed - predicted).
    method : string
      Method used to comput the standard deviations (i.e. dropout,
      heteroscedastic, etc.).
    figprefix : string
      String to prefix the filename to store the figure generated.
      A '_density_sigma_error.png' string will be appended to the
      figprefix given.
    """
    xbins = 51
    ybins = 31
    plt.figure(figsize=(24, 18))
    ax = plt.gca()
    plt.rc('xtick', labelsize=16)
    plt.hist2d(sigma, yerror, bins=[xbins, ybins], norm=LogNorm())
    cb = plt.colorbar()
    ax.set_xlabel('Standard Deviation (' + method + ')', fontsize=38,
        labelpad=15.0)
    ax.set_ylabel('Error: Observed - Mean Predicted', fontsize=38, labelpad
        =15.0)
    ax.axis([sigma.min() * 0.98, sigma.max() * 1.02, -yerror.max(), yerror.
        max()])
    plt.setp(ax.get_xticklabels(), fontsize=32)
    plt.setp(ax.get_yticklabels(), fontsize=32)
    cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=28)
    plt.grid(True)
    plt.savefig(figprefix + '_density_std_error.png', bbox_inches='tight')
    plt.close()
    print('Generated plot: ', figprefix + '_density_std_error.png')
