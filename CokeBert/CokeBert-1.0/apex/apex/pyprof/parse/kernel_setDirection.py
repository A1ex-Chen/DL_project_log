def setDirection(self):
    """
		Set direction (fprop, bprop) based on PyTorch sequence markers.
		It is a heuristic and not a foolproof method.
		"""
    if any('Backward, seq = ' in x for x in self.seqMarkers) or any(
        'backward, seq = ' in x for x in self.seqMarkers) or any(
        'Backward0, seq = ' in x for x in self.seqMarkers):
        self.dir = 'bprop'
    else:
        self.dir = 'fprop'
