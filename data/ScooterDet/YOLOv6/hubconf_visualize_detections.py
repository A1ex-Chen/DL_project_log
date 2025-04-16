def visualize_detections(image, boxes, classes, scores, min_score=0.4,
    figsize=(16, 16), linewidth=2, color='lawngreen'):
    image = np.array(image, dtype=np.uint8)
    fig = plt.figure(figsize=figsize)
    plt.axis('off')
    plt.imshow(image)
    ax = plt.gca()
    for box, name, score in zip(boxes, classes, scores):
        if score >= min_score:
            text = '{}: {:.2f}'.format(name, score)
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            patch = plt.Rectangle([x1, y1], w, h, fill=False, edgecolor=
                color, linewidth=linewidth)
            ax.add_patch(patch)
            ax.text(x1, y1, text, bbox={'facecolor': color, 'alpha': 0.8},
                clip_box=ax.clipbox, clip_on=True)
    plt.show()
