@staticmethod
def plot_2d_scatter_with_histogram(labels: List[str], label_to_x_data_dict:
    Dict[str, List], label_to_y_data_dict: Dict[str, List], title: str,
    on_pick_callback: Callable=None, label_to_pick_info_data_dict: Dict[str,
    List]=None):
    num_labels = len(labels)
    is_pickable = on_pick_callback is not None
    assert len(label_to_x_data_dict) == num_labels
    assert len(label_to_y_data_dict) == num_labels
    if is_pickable:
        assert label_to_pick_info_data_dict is not None and len(
            label_to_pick_info_data_dict) == num_labels
    all_x_data = [x for x_data in label_to_x_data_dict.values() for x in x_data
        ]
    all_y_data = [y for y_data in label_to_y_data_dict.values() for y in y_data
        ]
    grid = sns.jointplot(x=all_x_data, y=all_y_data, kind='reg', scatter=False)
    scatter_to_pick_info_data = {}
    for label in labels:
        x_data = label_to_x_data_dict[label]
        y_data = label_to_y_data_dict[label]
        scatter = grid.ax_joint.scatter(x=x_data, y=y_data, label=label,
            picker=is_pickable)
        if is_pickable:
            pick_info_data = label_to_pick_info_data_dict[label]
            scatter_to_pick_info_data[scatter] = pick_info_data
    grid.set_axis_labels(xlabel='Width', ylabel='Height')
    grid.ax_joint.set_title(title)
    grid.ax_joint.legend()
    fig = grid.fig
    fig.tight_layout()
    if is_pickable:

        def on_pick(event):
            index = event.ind[0]
            pick_info = scatter_to_pick_info_data[event.artist][index]
            on_pick_callback(pick_info)
        fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()
    plt.close()
