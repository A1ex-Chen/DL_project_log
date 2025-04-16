def plot_scatter(data, classes, out, width=10, height=8):
    cmap = plt.cm.get_cmap('gist_rainbow')
    plt.figure(figsize=(width, height))
    plt.scatter(data[:, 0], data[:, 1], c=classes, cmap=cmap, lw=0.5,
        edgecolor='black', alpha=0.7)
    plt.colorbar()
    png = '{}.png'.format(out)
    plt.savefig(png, bbox_inches='tight')
    plt.close()
