def is_kaggle():
    try:
        assert os.environ.get('PWD') == '/kaggle/working'
        assert os.environ.get('KAGGLE_URL_BASE') == 'https://www.kaggle.com'
        return True
    except AssertionError:
        return False
