def setOp(self):
    """
		Detect and set the class/module (mod) and operation (op)
		of the kernel e.g. torch.nn.functional / linear, torch / sigmoid.
		The lookup sequence we use is
			NVTX markers inserted by pyprof
			NVTX markers inserted by PyTorch in bprop
			NVTX markers inserted by PyTorch in fprop
		It is a heuristic and not a foolproof method.
		"""

    def sanitize(name):
        name = name.replace('torch', '').replace('autograd', '').replace(
            '_backward', '').replace('::', '').replace('jit', '').replace(
            '(anonymous namespace)', '')
        head, sep, tail = name.partition('Backward')
        return head
    for m in self.pyprofMarkers:
        assert 'mod' in m and 'op' in m and 'args' in m
        t = eval(m)
        self.op.append(t['op'])
        self.mod.append(t['mod'])
    if len(self.op):
        return
    for m in self.seqMarkers:
        if 'backward, seq = ' in m or 'Backward, seq = ' in m:
            op = m.split(',')[0]
            op = sanitize(op)
            self.op.append(op)
            self.mod.append('na')
    if len(self.op):
        return
    for m in self.seqMarkers:
        if ', seq = ' in m:
            op = m.split(',')[0]
            self.op.append(op)
            self.mod.append('na')
    if len(self.op):
        return
    if len(self.otherMarkers):
        self.op.append(self.otherMarkers[0])
    self.mod.append('na')
