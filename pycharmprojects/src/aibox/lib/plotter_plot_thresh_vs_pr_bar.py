@staticmethod
def plot_thresh_vs_pr_bar(num_classes: int, class_to_category_dict: Dict[
    int, str], class_to_ap_dict: Dict[int, float],
    class_to_recall_array_dict: Dict[int, np.ndarray],
    class_to_precision_array_dict: Dict[int, np.ndarray],
    class_to_f1_score_array_dict: Dict[int, np.ndarray],
    class_to_prob_array_dict: Dict[int, np.ndarray],
    path_to_placeholder_to_plot: str):

    def plot(plotting_recall_array_: np.ndarray, plotting_precision_array_:
        np.ndarray, plotting_f1_score_array_: np.ndarray,
        plotting_prob_array_: np.ndarray, category: str, ap: float,
        path_to_plot: str):
        matplotlib.use('agg')
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12.8, 7.68))
        bar_width = 0.2
        pos = np.arange(plotting_prob_array_.shape[0])
        ax.bar(pos - bar_width, plotting_precision_array_, bar_width, color
            ='blue', edgecolor='none')
        ax.bar(pos, plotting_recall_array_, bar_width, color='green',
            edgecolor='none')
        ax.bar(pos + bar_width, plotting_f1_score_array_, bar_width, color=
            'purple', edgecolor='none')
        for i, p in enumerate(pos):
            ax.text(p - bar_width, plotting_precision_array_[i] + 0.002,
                f'{plotting_precision_array_[i]:.3f}', color='blue',
                fontsize=6, rotation=90, ha='center', va='bottom')
            ax.text(p + 0.05, plotting_recall_array_[i] + 0.002,
                f'{plotting_recall_array_[i]:.3f}', color='green', fontsize
                =6, rotation=90, ha='center', va='bottom')
            ax.text(p + bar_width + 0.05, plotting_f1_score_array_[i] + 
                0.002, f'{plotting_f1_score_array_[i]:.3f}', color='purple',
                fontsize=6, rotation=90, ha='center', va='bottom')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        legends = ['Precision', 'Recall', 'F1-Score']
        ax.legend(legends, loc='upper left', bbox_to_anchor=(1, 1), shadow=True
            )
        ax.set_xlabel('Confidence Threshold')
        plt.xticks(pos, [f'{it:.4f}' for it in plotting_prob_array_],
            rotation=45)
        ax.set_ylim([0.0, 1.1])
        ax.set_title(f'Threshold versus PR: {category} AP = {ap:.4f}')
        ax.grid()
        fig.savefig(path_to_plot)
        plt.close()
    num_quantities = 20
    quantized_thresh_array = np.floor(np.linspace(1, 0, num_quantities + 1) *
        num_quantities) / num_quantities
    class_to_quantized_recall_array_dict = {}
    class_to_quantized_precision_array_dict = {}
    class_to_quantized_f1_score_array_dict = {}
    for c in range(1, num_classes):
        recall_array = class_to_recall_array_dict[c]
        precision_array = class_to_precision_array_dict[c]
        f1_score_array = class_to_f1_score_array_dict[c]
        prob_array = class_to_prob_array_dict[c]
        quantized_prob_array = np.floor(prob_array * num_quantities
            ) / num_quantities
        mask = (np.append(0, quantized_prob_array) != np.append(
            quantized_prob_array, 0))[1:]
        masked_recall_array = recall_array[mask]
        masked_precision_array = precision_array[mask]
        masked_f1_score_array = f1_score_array[mask]
        masked_prob_array = quantized_prob_array[mask]
        masked_recall_array = np.insert(masked_recall_array, 0, 0.0)
        masked_precision_array = np.insert(masked_precision_array, 0, 0.0)
        masked_f1_score_array = np.insert(masked_f1_score_array, 0, 0.0)
        masked_prob_array = np.insert(masked_prob_array, 0, 1.01)
        quantized_recall_array = []
        quantized_precision_array = []
        quantized_f1_score_array = []
        quantized_prob_array = []
        for thresh in quantized_thresh_array:
            idx = (masked_prob_array >= thresh).nonzero()[0][-1]
            quantized_recall_array.append(masked_recall_array[idx])
            quantized_precision_array.append(masked_precision_array[idx])
            quantized_f1_score_array.append(masked_f1_score_array[idx])
            quantized_prob_array.append(thresh)
        quantized_recall_array = np.array(quantized_recall_array)
        quantized_precision_array = np.array(quantized_precision_array)
        quantized_f1_score_array = np.array(quantized_f1_score_array)
        quantized_prob_array = np.array(quantized_prob_array)
        class_to_quantized_recall_array_dict[c] = quantized_recall_array
        class_to_quantized_precision_array_dict[c] = quantized_precision_array
        class_to_quantized_f1_score_array_dict[c] = quantized_f1_score_array
        plotting_recall_array = quantized_recall_array
        plotting_precision_array = quantized_precision_array
        plotting_f1_score_array = quantized_f1_score_array
        plotting_prob_array = quantized_prob_array
        if f1_score_array.shape[0] > 0:
            top_f1_score_index = f1_score_array.argmax().item()
            recall_at_top_f1_score = recall_array[top_f1_score_index]
            precision_at_top_f1_score = precision_array[top_f1_score_index]
            f1_score_at_top_f1_score = f1_score_array[top_f1_score_index]
            prob_at_top_f1_score = prob_array[top_f1_score_index]
            inserting_index = np.digitize(prob_at_top_f1_score,
                quantized_prob_array)
            plotting_recall_array = np.insert(plotting_recall_array,
                inserting_index, recall_at_top_f1_score)
            plotting_precision_array = np.insert(plotting_precision_array,
                inserting_index, precision_at_top_f1_score)
            plotting_f1_score_array = np.insert(plotting_f1_score_array,
                inserting_index, f1_score_at_top_f1_score)
            plotting_prob_array = np.insert(plotting_prob_array,
                inserting_index, prob_at_top_f1_score)
        plot(plotting_recall_array_=plotting_recall_array,
            plotting_precision_array_=plotting_precision_array,
            plotting_f1_score_array_=plotting_f1_score_array,
            plotting_prob_array_=plotting_prob_array, category=
            class_to_category_dict[c], ap=class_to_ap_dict[c], path_to_plot
            =path_to_placeholder_to_plot.format(c))
    plot(plotting_recall_array_=np.stack([
        class_to_quantized_recall_array_dict[c] for c in range(1,
        num_classes)], axis=0).mean(axis=0), plotting_precision_array_=np.
        stack([class_to_quantized_precision_array_dict[c] for c in range(1,
        num_classes)], axis=0).mean(axis=0), plotting_f1_score_array_=np.
        stack([class_to_quantized_f1_score_array_dict[c] for c in range(1,
        num_classes)], axis=0).mean(axis=0), plotting_prob_array_=
        quantized_thresh_array, category='mean', ap=np.mean([
        class_to_ap_dict[c] for c in range(1, num_classes)], axis=0).item(),
        path_to_plot=path_to_placeholder_to_plot.format('mean'))
