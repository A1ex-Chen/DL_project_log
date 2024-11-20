def plot_histogram_error_per_sigma(sigma, yerror, method=None, figprefix=None):
    """Functionality to plot a 1D histogram of the distribution of
       computed errors (i.e. values of observed - predicted) observed
       for specific values of standard deviations computed. The range of
       standard deviations computed is split in xbins values and the
       1D histograms of error distributions for the smallest six
       standard deviations are plotted.
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
      A '_histogram_error_per_sigma.png' string will be appended to
      the figprefix given.
    """
    xbins = 21
    ybins = 31
    H, xedges, yedges, img = plt.hist2d(sigma, yerror, bins=[xbins, ybins])
    plt.figure(figsize=(18, 24))
    legend = []
    for ii in range(4):
        if ii != 1:
            plt.plot(yedges[0:H.shape[1]], H[ii, :] / np.sum(H[ii, :]),
                marker='o', markersize=12, lw=6.0)
        legend.append(str((xedges[ii] + xedges[ii + 1]) / 2))
    plt.legend(legend, fontsize=28)
    ax = plt.gca()
    plt.title('Error Dist. per Standard Deviation for ' + method, fontsize=40)
    ax.set_xlabel('Error: Observed - Mean Predicted', fontsize=38, labelpad
        =15.0)
    ax.set_ylabel('Density', fontsize=38, labelpad=15.0)
    plt.setp(ax.get_xticklabels(), fontsize=32)
    plt.setp(ax.get_yticklabels(), fontsize=32)
    plt.grid(True)
    plt.savefig(figprefix + '_histogram_error_per_std.png', bbox_inches='tight'
        )
    plt.close()
    print('Generated plot: ', figprefix + '_histogram_error_per_std.png')
