def _diff_entities(ref_name, ref_doc, doc):
    ref_js = json.dumps(ref_doc, **DUMP_KWARGS)
    js = json.dumps(doc, **DUMP_KWARGS)
    return _diff_lines(ref_name, ref_js.splitlines(True), js.splitlines(True))
