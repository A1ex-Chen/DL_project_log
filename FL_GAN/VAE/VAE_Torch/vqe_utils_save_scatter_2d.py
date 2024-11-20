def save_scatter_2d(data, title):
    plt.figure()
    plt.title(title)
    plt.scatter(data[:, 0], data[:, 1])
