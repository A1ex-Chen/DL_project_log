def display_tracking_animation(detections, ground_truth_positions,
    predictions=None, width=360, height=240):

    def update(frame, detections, ground_truth_positions, predictions=None,
        width=360, height=240):
        ax.clear()
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_facecolor('white')
        for gt_pos in ground_truth_positions[frame]:
            x, y, w, h = gt_pos
            gt_box = patches.Rectangle((x, y), w, h, facecolor='blue',
                alpha=0.5)
            ax.add_patch(gt_box)
        for detection in detections[frame]:
            x, y, w, h = detection.box
            detection_box = patches.Rectangle((x, y), w, h, linewidth=1,
                edgecolor='green', facecolor='none')
            ax.add_patch(detection_box)
        if predictions is not None:
            for prediction in predictions[frame]:
                x, y, w, h = prediction
                prediction_box = patches.Rectangle((x, y), w, h, linewidth=
                    1, edgecolor='red', facecolor='none')
                ax.add_patch(prediction_box)
    if isinstance(detections[0], Detection):
        detections = [[detection] for detection in detections]
    if not isinstance(ground_truth_positions[0][0], list):
        ground_truth_positions = [[gt] for gt in ground_truth_positions]
    if predictions is not None and not isinstance(predictions[0][0], list):
        predictions = [[detection] for detection in predictions]
    fig, ax = plt.subplots(1, figsize=(12, 6))
    ani = FuncAnimation(fig, update, frames=len(detections), fargs=(
        detections, ground_truth_positions, predictions, width, height),
        interval=200)
    html = HTML(ani.to_jshtml())
    ipy_display(html)
    plt.close()
