def __init__(self, pos=None, text=None, color=None, font=QtGui.QFont()):
    GLGraphicsItem.__init__(self)
    self.color = color
    if color is None:
        self.color = QtCore.Qt.white
    self.text = text
    self.pos = pos
    self.font = font
    self.font.setPointSizeF(20)
