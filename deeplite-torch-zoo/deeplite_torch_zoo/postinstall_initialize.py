def initialize():
    try:
        import mmpretrain
    except ModuleNotFoundError:
        print('Initializing zoo...')
        install_mim_dependencies()
