def plot_array(nparray, xlabel, ylabel, title, fname):
    plt.figure()
    plt.plot(nparray, lw=3.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
