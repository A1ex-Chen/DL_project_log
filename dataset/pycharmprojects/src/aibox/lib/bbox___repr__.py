def __repr__(self) ->str:
    return ('BBox[l={:.1f}, t={:.1f}, r={:.1f}, b={:.1f}, w={:.1f}, h={:.1f}]'
        .format(self.left, self.top, self.right, self.bottom, self.width,
        self.height))
