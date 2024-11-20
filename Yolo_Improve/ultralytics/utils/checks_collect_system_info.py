def collect_system_info():
    """Collect and print relevant system information including OS, Python, RAM, CPU, and CUDA."""
    import psutil
    from ultralytics.utils import ENVIRONMENT, IS_GIT_DIR
    from ultralytics.utils.torch_utils import get_cpu_info
    ram_info = psutil.virtual_memory().total / 1024 ** 3
    check_yolo()
    LOGGER.info(
        f"""
{'OS':<20}{platform.platform()}
{'Environment':<20}{ENVIRONMENT}
{'Python':<20}{PYTHON_VERSION}
{'Install':<20}{'git' if IS_GIT_DIR else 'pip' if IS_PIP_PACKAGE else 'other'}
{'RAM':<20}{ram_info:.2f} GB
{'CPU':<20}{get_cpu_info()}
{'CUDA':<20}{torch.version.cuda if torch and torch.cuda.is_available() else None}
"""
        )
    for r in parse_requirements(package='ultralytics'):
        try:
            current = metadata.version(r.name)
            is_met = '✅ ' if check_version(current, str(r.specifier), hard=True
                ) else '❌ '
        except metadata.PackageNotFoundError:
            current = '(not installed)'
            is_met = '❌ '
        LOGGER.info(f'{r.name:<20}{is_met}{current}{r.specifier}')
    if is_github_action_running():
        LOGGER.info(
            f"""
RUNNER_OS: {os.getenv('RUNNER_OS')}
GITHUB_EVENT_NAME: {os.getenv('GITHUB_EVENT_NAME')}
GITHUB_WORKFLOW: {os.getenv('GITHUB_WORKFLOW')}
GITHUB_ACTOR: {os.getenv('GITHUB_ACTOR')}
GITHUB_REPOSITORY: {os.getenv('GITHUB_REPOSITORY')}
GITHUB_REPOSITORY_OWNER: {os.getenv('GITHUB_REPOSITORY_OWNER')}
"""
            )
