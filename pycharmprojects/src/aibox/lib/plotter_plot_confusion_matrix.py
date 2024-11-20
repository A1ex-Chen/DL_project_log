@staticmethod
def plot_confusion_matrix(confusion_matrix: np.ndarray,
    class_to_category_dict: Dict[int, str], path_to_plot: str):
    matplotlib.use('agg')
    categories = [v for k, v in class_to_category_dict.items() if k > 0]
    ax = sns.heatmap(data=confusion_matrix, cmap='YlGnBu', annot=True, fmt=
        'd', xticklabels=categories, yticklabels=categories)
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Ground Truth')
    fig = ax.get_figure()
    fig.savefig(path_to_plot)
    plt.close()
