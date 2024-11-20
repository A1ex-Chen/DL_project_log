def plot_history(out, history, metric='loss', val=True, title=None, width=8,
    height=6):
    title = title or 'model {}'.format(metric)
    val_metric = 'val_{}'.format(metric)
    plt.figure(figsize=(width, height))
    plt.plot(history.history[metric], marker='o')
    if val:
        plt.plot(history.history[val_metric], marker='d')
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    if val:
        plt.legend(['train_{}'.format(metric), 'val_{}'.format(metric)],
            loc='upper center')
    else:
        plt.legend(['train_{}'.format(metric)], loc='upper center')
    png = '{}.plot.{}.png'.format(out, metric)
    plt.savefig(png, bbox_inches='tight')
    plt.close()
