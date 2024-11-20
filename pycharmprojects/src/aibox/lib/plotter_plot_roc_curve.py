@staticmethod
def plot_roc_curve(num_classes: int, class_to_category_dict: Dict[int, str],
    macro_average_auc: float, class_to_auc_dict: Dict[int, float],
    class_to_fpr_array_dict: Dict[int, np.ndarray], class_to_tpr_array_dict:
    Dict[int, np.ndarray], path_to_plot: str):
    matplotlib.use('agg')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10.24, 7.68))
    for c in range(1, num_classes):
        fpr_array = class_to_fpr_array_dict[c]
        tpr_array = class_to_tpr_array_dict[c]
        ax.plot(fpr_array, tpr_array)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    category_list = [class_to_category_dict[c] for c in range(1, num_classes)]
    auc_list = [class_to_auc_dict[c] for c in range(1, num_classes)]
    legends = [f'{c}: {a:.4f}' for c, a in zip(category_list, auc_list)]
    ax.legend(legends, loc='upper left', bbox_to_anchor=(1, 1), shadow=True)
    plt.xticks(np.arange(0, 1.01, 0.1))
    plt.yticks(np.arange(0, 1.01, 0.1))
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_title(
        f"""Receiver Operating Characteristic Curve:
macro-average AUC = {macro_average_auc:.4f}"""
        )
    ax.grid()
    fig.savefig(path_to_plot)
    plt.close()
