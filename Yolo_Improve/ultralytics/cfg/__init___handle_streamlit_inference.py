def handle_streamlit_inference():
    """Open the Ultralytics Live Inference streamlit app for real time object detection."""
    checks.check_requirements(['streamlit', 'opencv-python', 'torch'])
    LOGGER.info('💡 Loading Ultralytics Live Inference app...')
    subprocess.run(['streamlit', 'run', ROOT /
        'solutions/streamlit_inference.py', '--server.headless', 'true'])
