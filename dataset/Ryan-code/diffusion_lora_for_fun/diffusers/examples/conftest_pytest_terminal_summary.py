def pytest_terminal_summary(terminalreporter):
    from diffusers.utils.testing_utils import pytest_terminal_summary_main
    make_reports = terminalreporter.config.getoption('--make-reports')
    if make_reports:
        pytest_terminal_summary_main(terminalreporter, id=make_reports)
