def _pltcolor_to_qtcolor(color):
    color_map = {'r': QtCore.Qt.red, 'g': QtCore.Qt.green, 'b': QtCore.Qt.
        blue, 'k': QtCore.Qt.black, 'w': QtCore.Qt.white, 'y': QtCore.Qt.
        yellow, 'c': QtCore.Qt.cyan, 'm': QtCore.Qt.magenta}
    return color_map[color]
