def plot(self, annotations, output, bbox=None, points=None, point_label=
    None, mask_random_color=True, better_quality=True, retina=False,
    with_contours=True):
    """
        Plots annotations, bounding boxes, and points on images and saves the output.

        Args:
            annotations (list): Annotations to be plotted.
            output (str or Path): Output directory for saving the plots.
            bbox (list, optional): Bounding box coordinates [x1, y1, x2, y2]. Defaults to None.
            points (list, optional): Points to be plotted. Defaults to None.
            point_label (list, optional): Labels for the points. Defaults to None.
            mask_random_color (bool, optional): Whether to use random color for masks. Defaults to True.
            better_quality (bool, optional): Whether to apply morphological transformations for better mask quality.
                Defaults to True.
            retina (bool, optional): Whether to use retina mask. Defaults to False.
            with_contours (bool, optional): Whether to plot contours. Defaults to True.
        """
    import matplotlib.pyplot as plt
    pbar = TQDM(annotations, total=len(annotations))
    for ann in pbar:
        result_name = os.path.basename(ann.path)
        image = ann.orig_img[..., ::-1]
        original_h, original_w = ann.orig_shape
        plt.figure(figsize=(original_w / 100, original_h / 100))
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0,
            wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.imshow(image)
        if ann.masks is not None:
            masks = ann.masks.data
            if better_quality:
                if isinstance(masks[0], torch.Tensor):
                    masks = np.array(masks.cpu())
                for i, mask in enumerate(masks):
                    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.
                        MORPH_CLOSE, np.ones((3, 3), np.uint8))
                    masks[i] = cv2.morphologyEx(mask.astype(np.uint8), cv2.
                        MORPH_OPEN, np.ones((8, 8), np.uint8))
            self.fast_show_mask(masks, plt.gca(), random_color=
                mask_random_color, bbox=bbox, points=points, pointlabel=
                point_label, retinamask=retina, target_height=original_h,
                target_width=original_w)
            if with_contours:
                contour_all = []
                temp = np.zeros((original_h, original_w, 1))
                for i, mask in enumerate(masks):
                    mask = mask.astype(np.uint8)
                    if not retina:
                        mask = cv2.resize(mask, (original_w, original_h),
                            interpolation=cv2.INTER_NEAREST)
                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2
                        .CHAIN_APPROX_SIMPLE)
                    contour_all.extend(iter(contours))
                cv2.drawContours(temp, contour_all, -1, (255, 255, 255), 2)
                color = np.array([0 / 255, 0 / 255, 1.0, 0.8])
                contour_mask = temp / 255 * color.reshape(1, 1, -1)
                plt.imshow(contour_mask)
        save_path = Path(output) / result_name
        save_path.parent.mkdir(exist_ok=True, parents=True)
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0,
            transparent=True)
        plt.close()
        pbar.set_description(f'Saving {result_name} to {save_path}')
