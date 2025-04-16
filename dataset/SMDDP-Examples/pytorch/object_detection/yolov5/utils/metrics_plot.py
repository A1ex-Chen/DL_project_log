def plot(self, normalize=True, save_dir='', names=()):
    try:
        import seaborn as sn
        array = self.matrix / (self.matrix.sum(0).reshape(1, -1) + 1e-09 if
            normalize else 1)
        array[array < 0.005] = np.nan
        fig = plt.figure(figsize=(12, 9), tight_layout=True)
        nc, nn = self.nc, len(names)
        sn.set(font_scale=1.0 if nc < 50 else 0.8)
        labels = 0 < nn < 99 and nn == nc
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sn.heatmap(array, annot=nc < 30, annot_kws={'size': 8}, cmap=
                'Blues', fmt='.2f', square=True, vmin=0.0, xticklabels=
                names + ['background FP'] if labels else 'auto',
                yticklabels=names + ['background FN'] if labels else 'auto'
                ).set_facecolor((1, 1, 1))
        fig.axes[0].set_xlabel('True')
        fig.axes[0].set_ylabel('Predicted')
        plt.title('Confusion Matrix')
        fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        plt.close()
    except Exception as e:
        print(f'WARNING: ConfusionMatrix plot failure: {e}')
