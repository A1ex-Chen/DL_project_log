def plot(self, save_dir='', names=()):
    try:
        import seaborn as sn
        array = self.matrix / (self.matrix.sum(0).reshape(1, self.nc + 1) +
            1e-06)
        array[array < 0.005] = np.nan
        fig = plt.figure(figsize=(12, 9), tight_layout=True)
        sn.set(font_scale=1.0 if self.nc < 50 else 0.8)
        labels = 0 < len(names) < 99 and len(names) == self.nc
        sn.heatmap(array, annot=self.nc < 30, annot_kws={'size': 8}, cmap=
            'Blues', fmt='.2f', square=True, xticklabels=names + [
            'background FP'] if labels else 'auto', yticklabels=names + [
            'background FN'] if labels else 'auto').set_facecolor((1, 1, 1))
        fig.axes[0].set_xlabel('True')
        fig.axes[0].set_ylabel('Predicted')
        fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
    except Exception as e:
        pass
