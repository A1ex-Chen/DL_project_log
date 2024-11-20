@staticmethod
def fast_show_mask(annotation, ax, random_color=False, bbox=None, points=
    None, pointlabel=None, retinamask=True, target_height=960, target_width=960
    ):
    """
        Quickly shows the mask annotations on the given matplotlib axis.

        Args:
            annotation (array-like): Mask annotation.
            ax (matplotlib.axes.Axes): Matplotlib axis.
            random_color (bool, optional): Whether to use random color for masks. Defaults to False.
            bbox (list, optional): Bounding box coordinates [x1, y1, x2, y2]. Defaults to None.
            points (list, optional): Points to be plotted. Defaults to None.
            pointlabel (list, optional): Labels for the points. Defaults to None.
            retinamask (bool, optional): Whether to use retina mask. Defaults to True.
            target_height (int, optional): Target height for resizing. Defaults to 960.
            target_width (int, optional): Target width for resizing. Defaults to 960.
        """
    import matplotlib.pyplot as plt
    n, h, w = annotation.shape
    areas = np.sum(annotation, axis=(1, 2))
    annotation = annotation[np.argsort(areas)]
    index = (annotation != 0).argmax(axis=0)
    if random_color:
        color = np.random.random((n, 1, 1, 3))
    else:
        color = np.ones((n, 1, 1, 3)) * np.array([30 / 255, 144 / 255, 1.0])
    transparency = np.ones((n, 1, 1, 1)) * 0.6
    visual = np.concatenate([color, transparency], axis=-1)
    mask_image = np.expand_dims(annotation, -1) * visual
    show = np.zeros((h, w, 4))
    h_indices, w_indices = np.meshgrid(np.arange(h), np.arange(w), indexing
        ='ij')
    indices = index[h_indices, w_indices], h_indices, w_indices, slice(None)
    show[h_indices, w_indices, :] = mask_image[indices]
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
            edgecolor='b', linewidth=1))
    if points is not None:
        plt.scatter([point[0] for i, point in enumerate(points) if 
            pointlabel[i] == 1], [point[1] for i, point in enumerate(points
            ) if pointlabel[i] == 1], s=20, c='y')
        plt.scatter([point[0] for i, point in enumerate(points) if 
            pointlabel[i] == 0], [point[1] for i, point in enumerate(points
            ) if pointlabel[i] == 0], s=20, c='m')
    if not retinamask:
        show = cv2.resize(show, (target_width, target_height),
            interpolation=cv2.INTER_NEAREST)
    ax.imshow(show)
