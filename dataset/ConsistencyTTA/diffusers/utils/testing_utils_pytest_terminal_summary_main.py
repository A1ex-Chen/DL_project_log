def pytest_terminal_summary_main(tr, id):
    """
    Generate multiple reports at the end of test suite run - each report goes into a dedicated file in the current
    directory. The report files are prefixed with the test suite name.

    This function emulates --duration and -rA pytest arguments.

    This function is to be called from `conftest.py` via `pytest_terminal_summary` wrapper that has to be defined
    there.

    Args:
    - tr: `terminalreporter` passed from `conftest.py`
    - id: unique id like `tests` or `examples` that will be incorporated into the final reports filenames - this is
      needed as some jobs have multiple runs of pytest, so we can't have them overwrite each other.

    NB: this functions taps into a private _pytest API and while unlikely, it could break should
    pytest do internal changes - also it calls default internal methods of terminalreporter which
    can be hijacked by various `pytest-` plugins and interfere.

    """
    from _pytest.config import create_terminal_writer
    if not len(id):
        id = 'tests'
    config = tr.config
    orig_writer = config.get_terminal_writer()
    orig_tbstyle = config.option.tbstyle
    orig_reportchars = tr.reportchars
    dir = 'reports'
    Path(dir).mkdir(parents=True, exist_ok=True)
    report_files = {k: f'{dir}/{id}_{k}.txt' for k in ['durations',
        'errors', 'failures_long', 'failures_short', 'failures_line',
        'passes', 'stats', 'summary_short', 'warnings']}
    dlist = []
    for replist in tr.stats.values():
        for rep in replist:
            if hasattr(rep, 'duration'):
                dlist.append(rep)
    if dlist:
        dlist.sort(key=lambda x: x.duration, reverse=True)
        with open(report_files['durations'], 'w') as f:
            durations_min = 0.05
            f.write('slowest durations\n')
            for i, rep in enumerate(dlist):
                if rep.duration < durations_min:
                    f.write(
                        f'{len(dlist) - i} durations < {durations_min} secs were omitted'
                        )
                    break
                f.write(f'{rep.duration:02.2f}s {rep.when:<8} {rep.nodeid}\n')

    def summary_failures_short(tr):
        reports = tr.getreports('failed')
        if not reports:
            return
        tr.write_sep('=', 'FAILURES SHORT STACK')
        for rep in reports:
            msg = tr._getfailureheadline(rep)
            tr.write_sep('_', msg, red=True, bold=True)
            longrepr = re.sub('.*_ _ _ (_ ){10,}_ _ ', '', rep.longreprtext,
                0, re.M | re.S)
            tr._tw.line(longrepr)
    config.option.tbstyle = 'auto'
    with open(report_files['failures_long'], 'w') as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_failures()
    with open(report_files['failures_short'], 'w') as f:
        tr._tw = create_terminal_writer(config, f)
        summary_failures_short(tr)
    config.option.tbstyle = 'line'
    with open(report_files['failures_line'], 'w') as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_failures()
    with open(report_files['errors'], 'w') as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_errors()
    with open(report_files['warnings'], 'w') as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_warnings()
        tr.summary_warnings()
    tr.reportchars = 'wPpsxXEf'
    with open(report_files['passes'], 'w') as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_passes()
    with open(report_files['summary_short'], 'w') as f:
        tr._tw = create_terminal_writer(config, f)
        tr.short_test_summary()
    with open(report_files['stats'], 'w') as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_stats()
    tr._tw = orig_writer
    tr.reportchars = orig_reportchars
    config.option.tbstyle = orig_tbstyle
