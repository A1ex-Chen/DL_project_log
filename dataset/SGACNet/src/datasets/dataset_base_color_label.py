def color_label(self, label, with_void=True):
    if with_void:
        colors = self.class_colors
    else:
        colors = self.class_colors_without_void
    cmap = np.asarray(colors, dtype='uint8')
    return cmap[label]
