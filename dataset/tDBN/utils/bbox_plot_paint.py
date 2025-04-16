def paint(self):
    self.GLViewWidget.qglColor(self.color)
    if self.pos is not None and self.text is not None:
        if isinstance(self.pos, (list, tuple, np.ndarray)):
            for p, text in zip(self.pos, self.text):
                self.GLViewWidget.renderText(*p, text, self.font)
        else:
            self.GLViewWidget.renderText(*self.pos, self.text, self.font)
