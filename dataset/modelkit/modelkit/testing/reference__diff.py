def _diff(self, ref_name, ref, doc):
    doc = _ensure_lines(doc)
    lines = [(line + '\n') for line in doc]
    _diff_lines(ref_name, ref.splitlines(True), lines)
