def _diff_lines(ref_name, ref_lines, lines):
    if ref_lines == lines:
        return
    diff = list(difflib.unified_diff(ref_lines, lines, fromfile=ref_name,
        tofile='test output'))
    diff = ''.join(diff)
    assert False, diff
