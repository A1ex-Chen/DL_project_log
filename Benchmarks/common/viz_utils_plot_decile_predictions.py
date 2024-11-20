def plot_decile_predictions(Ypred, Ypred_Lp, Ypred_Hp, decile_list,
    pred_name=None, figprefix=None):
    """Functionality to plot the mean of the deciles predicted.
       The plot generated is stored in a png file.

    Parameters
    ----------
    Ypred : numpy array
      Array with median predicted values.
    Ypred_Lp : numpy array
      Array with low decile predicted values.
    Ypred_Hp : numpy array
      Array with high decile predicted values.
    decile_list : string list
      List of deciles predicted (e.g. '1st', '9th', etc.)
    pred_name : string
      Name of data colum or quantity predicted (e.g. growth, AUC, etc.)
    figprefix : string
      String to prefix the filename to store the figure generated.
      A '_decile_predictions.png' string will be appended to the
      figprefix given.
    """
    index_ = np.argsort(Ypred)
    plt.figure(figsize=(24, 18))
    plt.scatter(range(index_.shape[0]), Ypred[index_])
    plt.scatter(range(index_.shape[0]), Ypred_Lp[index_])
    plt.scatter(range(index_.shape[0]), Ypred_Hp[index_])
    plt.legend(decile_list, fontsize=28)
    plt.xlabel('Index', fontsize=38.0)
    plt.ylabel(pred_name, fontsize=38.0)
    plt.title('Predicted ' + pred_name + ' Deciles', fontsize=40)
    plt.grid()
    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), fontsize=32)
    plt.setp(ax.get_yticklabels(), fontsize=32)
    plt.savefig(figprefix + '_decile_predictions.png', bbox_inches='tight')
    plt.close()
    print('Generated plot: ', figprefix + '_decile_predictions.png')
