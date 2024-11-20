def find_sub_list(self, sl, l):
    sll = len(sl)
    if sll > 0:
        for ind in (i for i, e in enumerate(l) if e == sl[0]):
            if l[ind:ind + sll] == sl:
                return ind, ind + sll - 1
    return 0, 1
