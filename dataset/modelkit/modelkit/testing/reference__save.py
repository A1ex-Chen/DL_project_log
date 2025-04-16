def _save(self, doc, fp):
    doc = _ensure_lines(doc)
    for line in doc:
        print(line.rstrip('\n'), file=fp)
