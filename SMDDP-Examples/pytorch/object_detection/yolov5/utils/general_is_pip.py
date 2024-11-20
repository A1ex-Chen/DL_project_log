def is_pip():
    return 'site-packages' in Path(__file__).resolve().parts
