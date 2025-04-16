def handle_explorer():
    """Open the Ultralytics Explorer GUI for dataset exploration and analysis."""
    checks.check_requirements('streamlit')
    LOGGER.info('💡 Loading Explorer dashboard...')
    subprocess.run(['streamlit', 'run', ROOT / 'data/explorer/gui/dash.py',
        '--server.maxMessageSize', '2048'])
