def plot_evolution(yaml_file='data/hyp.finetune.yaml'):
    with open(yaml_file) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)
    x = np.loadtxt('evolve.txt', ndmin=2)
    f = fitness(x)
    plt.figure(figsize=(10, 12), tight_layout=True)
    matplotlib.rc('font', **{'size': 8})
    for i, (k, v) in enumerate(hyp.items()):
        y = x[:, i + 7]
        mu = y[f.argmax()]
        plt.subplot(6, 5, i + 1)
        plt.scatter(y, f, c=hist2d(y, f, 20), cmap='viridis', alpha=0.8,
            edgecolors='none')
        plt.plot(mu, f.max(), 'k+', markersize=15)
        plt.title('%s = %.3g' % (k, mu), fontdict={'size': 9})
        if i % 5 != 0:
            plt.yticks([])
        print('%15s: %.3g' % (k, mu))
    plt.savefig('evolve.png', dpi=200)
    print('\nPlot saved as evolve.png')
