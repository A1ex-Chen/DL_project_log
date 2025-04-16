@staticmethod
def static_color_label(label, colors):
    cmap = np.asarray(colors, dtype='uint8')
    return cmap[label]
