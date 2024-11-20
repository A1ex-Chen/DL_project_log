def add_boxes(boxes, ax, scores=None, class_ids=None, class_names=None,
    threshold=0):
    if threshold > 0:
        assert scores != None
        N = tf.where(scores > threshold).shape[0]
    else:
        N = boxes.shape[0]
    if not N:
        print('\n*** No instances to display *** \n')
        return ax
    colors = random_colors(N)
    for i in range(N):
        color = colors[i]
        if not np.any(boxes[i]):
            continue
        y1, x1, y2, x2 = boxes[i]
        y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
            alpha=0.7, linestyle='dashed', edgecolor=color, facecolor='none')
        ax.add_patch(p)
        if scores != None:
            class_id = int(class_ids[i])
            score = scores[i] if scores is not None else None
            label = class_names[class_id
                ] if class_names is not None else class_id
            x = random.randint(x1, (x1 + x2) // 2)
            caption = '{} {:.3f}'.format(label, score) if score else label
            ax.text(x1, y1 + 8, caption, size=15, color=color,
                backgroundcolor='none')
    return ax
