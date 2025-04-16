def draw_bbox_on_latest_picture(self, response, history=None) ->Optional[Image
    .Image]:
    image = self._fetch_latest_picture(response, history)
    if image is None:
        return None
    if image.startswith('http://') or image.startswith('https://'):
        image = Image.open(requests.get(image, stream=True).raw).convert('RGB')
        h, w = image.height, image.width
    else:
        image = np.asarray(Image.open(image).convert('RGB'))
        h, w = image.shape[0], image.shape[1]
    visualizer = Visualizer(image)
    boxes = self._fetch_all_box_with_ref(response)
    if not boxes:
        return None
    color = random.choice([_ for _ in mcolors.TABLEAU_COLORS.keys()])
    for box in boxes:
        if 'ref' in box:
            color = random.choice([_ for _ in mcolors.TABLEAU_COLORS.keys()])
        x1, y1, x2, y2 = box['box']
        x1, y1, x2, y2 = int(x1 / 1000 * w), int(y1 / 1000 * h), int(x2 / 
            1000 * w), int(y2 / 1000 * h)
        visualizer.draw_box((x1, y1, x2, y2), alpha=1, edge_color=color)
        if 'ref' in box:
            visualizer.draw_text(box['ref'], (x1, y1), color=color,
                horizontal_alignment='left')
    return visualizer.output
