def update(frame, detections, ground_truth_positions, predictions=None,
    width=360, height=240):
    ax.clear()
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_facecolor('white')
    for gt_pos in ground_truth_positions[frame]:
        x, y, w, h = gt_pos
        gt_box = patches.Rectangle((x, y), w, h, facecolor='blue', alpha=0.5)
        ax.add_patch(gt_box)
    for detection in detections[frame]:
        x, y, w, h = detection.box
        detection_box = patches.Rectangle((x, y), w, h, linewidth=1,
            edgecolor='green', facecolor='none')
        ax.add_patch(detection_box)
    if predictions is not None:
        for prediction in predictions[frame]:
            x, y, w, h = prediction
            prediction_box = patches.Rectangle((x, y), w, h, linewidth=1,
                edgecolor='red', facecolor='none')
            ax.add_patch(prediction_box)
