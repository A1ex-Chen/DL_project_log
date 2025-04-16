from typing import List, Dict, Tuple, Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class Plotter:

    @staticmethod

    @staticmethod

    @staticmethod

    @staticmethod

    @staticmethod

    @staticmethod

    @staticmethod

        num_quantities = 20
        quantized_thresh_array = np.floor(np.linspace(1, 0, num_quantities + 1) * num_quantities) / num_quantities

        class_to_quantized_recall_array_dict = {}
        class_to_quantized_precision_array_dict = {}
        class_to_quantized_f1_score_array_dict = {}

        for c in range(1, num_classes):
            recall_array = class_to_recall_array_dict[c]
            precision_array = class_to_precision_array_dict[c]
            f1_score_array = class_to_f1_score_array_dict[c]
            prob_array = class_to_prob_array_dict[c]

            # NOTE: Quantize into `num_quantities` bins from 0 to 1,
            #           for example:
            #               num_quantities = 20
            #               prob_array = 0.732  0.675  0.653  0.621  0.531  0.519
            #     quantized_prob_array = 0.70   0.65   0.65   0.60   0.50   0.50
            quantized_prob_array = np.floor(prob_array * num_quantities) / num_quantities

            # NOTE: Example as below
            #
            #                  prob_array =        0.70   0.65   0.65   0.60   0.50   0.50
            #                recall_array =        0.124  0.336  0.381  0.433  0.587  0.590
            #             precision_array =        0.883  0.707  0.733  0.684  0.512  0.506
            #              f1_score_array =        0.217  0.456  0.501  0.530  0.547  0.545
            #
            #    np.append(0, prob_array) = 0.00   0.70   0.65   0.65   0.60   0.50   0.50
            #    np.append(prob_array, 0) = 0.70   0.65   0.65   0.60   0.50   0.50   0.00
            #             unequal compare =    T      T      F      F      T      F      T
            #                        mask =           T      F      F      T      F      T
            #
            #            prob_array[mask] =        0.70                 0.60          0.50
            #          recall_array[mask] =        0.124                0.433         0.590
            #       precision_array[mask] =        0.883                0.684         0.506
            #        f1_score_array[mask] =        0.217                0.530         0.545
            #
            #    result keep only if [n]-th element is not equal to [n+1]-th element
            mask = (np.append(0, quantized_prob_array) != np.append(quantized_prob_array, 0))[1:]
            masked_recall_array = recall_array[mask]
            masked_precision_array = precision_array[mask]
            masked_f1_score_array = f1_score_array[mask]
            masked_prob_array = quantized_prob_array[mask]

            masked_recall_array = np.insert(masked_recall_array, 0, 0.)
            masked_precision_array = np.insert(masked_precision_array, 0, 0.)
            masked_f1_score_array = np.insert(masked_f1_score_array, 0, 0.)
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

            # NOTE: The metric point at top f1-score is useful information when choosing a threshold for the specific class;
            #       however, it can be lost after quantization (in the above example, the best f1-score 0.547 was gone),
            #       hence we are going to put it back here
            if f1_score_array.shape[0] > 0:
                top_f1_score_index = f1_score_array.argmax().item()
                recall_at_top_f1_score = recall_array[top_f1_score_index]
                precision_at_top_f1_score = precision_array[top_f1_score_index]
                f1_score_at_top_f1_score = f1_score_array[top_f1_score_index]
                prob_at_top_f1_score = prob_array[top_f1_score_index]

                inserting_index = np.digitize(prob_at_top_f1_score, quantized_prob_array)
                plotting_recall_array = np.insert(plotting_recall_array, inserting_index, recall_at_top_f1_score)
                plotting_precision_array = np.insert(plotting_precision_array, inserting_index, precision_at_top_f1_score)
                plotting_f1_score_array = np.insert(plotting_f1_score_array, inserting_index, f1_score_at_top_f1_score)
                plotting_prob_array = np.insert(plotting_prob_array, inserting_index, prob_at_top_f1_score)

            plot(plotting_recall_array_=plotting_recall_array,
                 plotting_precision_array_=plotting_precision_array,
                 plotting_f1_score_array_=plotting_f1_score_array,
                 plotting_prob_array_=plotting_prob_array,
                 category=class_to_category_dict[c],
                 ap=class_to_ap_dict[c],
                 path_to_plot=path_to_placeholder_to_plot.format(c))

        plot(
            plotting_recall_array_=np.stack([class_to_quantized_recall_array_dict[c] for c in range(1, num_classes)], axis=0).mean(axis=0),
            plotting_precision_array_=np.stack([class_to_quantized_precision_array_dict[c] for c in range(1, num_classes)], axis=0).mean(axis=0),
            plotting_f1_score_array_=np.stack([class_to_quantized_f1_score_array_dict[c] for c in range(1, num_classes)], axis=0).mean(axis=0),
            plotting_prob_array_=quantized_thresh_array,
            category='mean',
            ap=np.mean([class_to_ap_dict[c] for c in range(1, num_classes)], axis=0).item(),
            path_to_plot=path_to_placeholder_to_plot.format('mean')
        )

    @staticmethod
    def plot_2d_scatter_with_histogram(labels: List[str],
                                       label_to_x_data_dict: Dict[str, List],
                                       label_to_y_data_dict: Dict[str, List],
                                       title: str,
                                       on_pick_callback: Callable = None,
                                       label_to_pick_info_data_dict: Dict[str, List] = None):
        num_labels = len(labels)
        is_pickable = on_pick_callback is not None

        assert len(label_to_x_data_dict) == num_labels
        assert len(label_to_y_data_dict) == num_labels
        if is_pickable:
            assert label_to_pick_info_data_dict is not None and len(label_to_pick_info_data_dict) == num_labels

        all_x_data = [x for x_data in label_to_x_data_dict.values() for x in x_data]
        all_y_data = [y for y_data in label_to_y_data_dict.values() for y in y_data]
        grid = sns.jointplot(x=all_x_data, y=all_y_data, kind='reg', scatter=False)

        scatter_to_pick_info_data = {}
        for label in labels:
            x_data = label_to_x_data_dict[label]
            y_data = label_to_y_data_dict[label]
            scatter = grid.ax_joint.scatter(x=x_data, y=y_data, label=label, picker=is_pickable)
            if is_pickable:
                pick_info_data = label_to_pick_info_data_dict[label]
                scatter_to_pick_info_data[scatter] = pick_info_data

        grid.set_axis_labels(xlabel='Width', ylabel='Height')
        grid.ax_joint.set_title(title)
        grid.ax_joint.legend()
        fig = grid.fig
        fig.tight_layout()

        if is_pickable:
            fig.canvas.mpl_connect('pick_event', on_pick)

        plt.show()
        plt.close()

    @staticmethod
    def plot_category_vs_count_bar(category_vs_count_dict: Dict[str, int]):
        categories = [k for k in category_vs_count_dict.keys()]
        counts = [v for v in category_vs_count_dict.values()]
        category_and_count_list = [(category, count)
                                   for count, category in sorted(zip(counts, categories), reverse=True)]

        ax = sns.barplot(x=[category for category, _ in category_and_count_list],
                         y=[count for _, count in category_and_count_list])
        for patch in ax.patches:
            ax.annotate(f'{int(patch.get_height())}',
                        (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points')
        ax.set_xlabel('Category')
        ax.set_ylabel('Count')
        fig = ax.get_figure()
        fig.tight_layout()

        plt.show()
        plt.close()