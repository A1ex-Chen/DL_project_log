@TryExcept('WARNING ⚠️ ConfusionMatrix plot failure')
def plot(self, normalize=True, save_dir='', names=()):
    import seaborn as sn
    array = self.matrix / (self.matrix.sum(0).reshape(1, -1) + 1e-09 if
        normalize else 1)
    array[array < 0.005] = np.nan
    fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
    nc, nn = self.nc, len(names)
    sn.set(font_scale=1.0 if nc < 50 else 0.8)
    labels = 0 < nn < 99 and nn == nc
    ticklabels = names + ['background'] if labels else 'auto'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sn.heatmap(array, ax=ax, annot=nc < 30, annot_kws={'size': 8}, cmap
            ='Blues', fmt='.2f', square=True, vmin=0.0, xticklabels=
            ticklabels, yticklabels=ticklabels).set_facecolor((1, 1, 1))
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.set_title('Confusion Matrix')
    fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
    plt.close(fig)
