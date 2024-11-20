def save_indexed_png(filepath, label, colormap):
    img = Image.fromarray(np.asarray(label, dtype='uint8'))
    img.putpalette(list(np.asarray(colormap, dtype='uint8').flatten()))
    img.save(filepath, 'PNG')
