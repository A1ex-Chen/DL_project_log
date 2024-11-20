def summary_failures_short(tr):
    reports = tr.getreports('failed')
    if not reports:
        return
    tr.write_sep('=', 'FAILURES SHORT STACK')
    for rep in reports:
        msg = tr._getfailureheadline(rep)
        tr.write_sep('_', msg, red=True, bold=True)
        longrepr = re.sub('.*_ _ _ (_ ){10,}_ _ ', '', rep.longreprtext, 0,
            re.M | re.S)
        tr._tw.line(longrepr)
