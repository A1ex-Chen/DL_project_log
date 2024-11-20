def __init__(self):
    """Initializes the Events object with default values for events, rate_limit, and metadata."""
    self.events = []
    self.rate_limit = 60.0
    self.t = 0.0
    self.metadata = {'cli': Path(ARGV[0]).name == 'yolo', 'install': 'git' if
        IS_GIT_DIR else 'pip' if IS_PIP_PACKAGE else 'other', 'python': '.'
        .join(platform.python_version_tuple()[:2]), 'version': __version__,
        'env': ENVIRONMENT, 'session_id': round(random.random() * 
        1000000000000000.0), 'engagement_time_msec': 1000}
    self.enabled = SETTINGS['sync'] and RANK in {-1, 0
        } and not TESTS_RUNNING and ONLINE and (IS_PIP_PACKAGE or 
        get_git_origin_url() ==
        'https://github.com/ultralytics/ultralytics.git')
