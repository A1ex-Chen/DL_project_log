def check_pre(a, b):
    if len(a) < len(b):
        return False
    else:
        a = [x for x in a if x != 'ĠUCHIJ' and x != 'ĠTG' and x != 'ĠUKIP' and
            x != 'ĠCLSID']
        for i in range(len(b)):
            if a[i] != b[i]:
                return False
        return True
