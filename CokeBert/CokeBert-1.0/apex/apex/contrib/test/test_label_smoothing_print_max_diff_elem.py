def print_max_diff_elem(self, ref, tst):
    ref, tst = ref.flatten(), tst.flatten()
    diff = (ref - tst).abs().max()
    idx = (ref - tst).abs().argmax()
    print('Max atol idx: {}, diff: {:.6f}, ref: {:.6f}, tst: {:.6f}'.format
        (idx, diff, ref[idx], tst[idx]))
