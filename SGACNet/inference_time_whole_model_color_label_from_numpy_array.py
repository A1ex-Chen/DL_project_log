def color_label_from_numpy_array(label):
    cmap = np.asarray(SUNRBDBase.CLASS_COLORS, dtype='uint8')
    return cmap[label]
