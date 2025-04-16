def is_kaggle():
    return os.environ.get('PWD') == '/kaggle/working' and os.environ.get(
        'KAGGLE_URL_BASE') == 'https://www.kaggle.com'
