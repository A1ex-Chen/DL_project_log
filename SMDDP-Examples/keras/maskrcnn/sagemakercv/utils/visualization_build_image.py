def build_image(image, boxes=None, scores=None, class_ids=None, class_names
    =None, threshold=0, figsize=(10, 10), title=''):
    if tf.is_tensor(image):
        image = image.numpy()
    fig, ax = plt.subplots(1, figsize=figsize)
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)
    if boxes != None:
        ax = add_boxes(boxes, ax, scores, class_ids, class_names, threshold)
    plt.imshow(image.astype(np.uint8))
    fig.tight_layout(pad=0.0)
    fig.canvas.draw()
    plt.close(fig)
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data
