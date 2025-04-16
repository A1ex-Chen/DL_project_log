def fancy_plot(y, color='C0', label='', alpha=0.3):
    """
    A function for a nice visualization of MRLT.
    """
    n = y.shape[0]
    x = np.arange(n)
    xleft = x - 0.5
    xright = x + 0.5
    X = np.array([xleft, xright]).T.flatten()
    Xn = np.zeros(X.shape[0] + 2)
    Xn[1:-1] = X
    Xn[0] = -0.5
    Xn[-1] = n - 0.5
    Y = np.array([y, y]).T.flatten()
    Yn = np.zeros(Y.shape[0] + 2)
    Yn[1:-1] = Y
    plt.bar(x, y, width=1, alpha=alpha, color=color, edgecolor=color)
    plt.plot(Xn, Yn, c=color, label=label, lw=3)
